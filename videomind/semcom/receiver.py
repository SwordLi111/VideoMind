# Receiver-side LLM (frozen Qwen2.5-7B-Instruct) that reads prompts mixing plain text and soft tokens.
#
# A prompt is a list of segments, each either a string or a tensor of soft tokens with shape [K, D]. The same prompt
# builders are used with plain caption strings (noiseless text baseline) and with decoded soft tokens (semantic
# communication), so that the two settings only differ in how each clip description reaches the receiver.

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from videomind.eval.infer_caption import SCORE_INSTRUCTIONS

MARKER = '<<CONTENT>>'


class Receiver:

    def __init__(self, model_path, device='cuda', gradient_checkpointing=False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
        self.model.requires_grad_(False)
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=dict(use_reentrant=False))
            # checkpointing is only active in train mode, which is safe since all dropout rates of Qwen2.5 are 0
            assert self.model.config.attention_dropout == 0
            self.model.train()
        self.device = device
        self.embed = self.model.get_input_embeddings()

        # chat template around the user message, split at the content
        text = self.tokenizer.apply_chat_template([{
            'role': 'user',
            'content': MARKER
        }],
                                                  tokenize=False,
                                                  add_generation_prompt=True)
        self.chat_head, self.chat_tail = text.split(MARKER)

    def _ids(self, text):
        return self.tokenizer(text, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device)

    def embed_segments(self, segments):
        parts = []
        for seg in segments:
            if isinstance(seg, str):
                if len(seg) > 0:
                    parts.append(self.embed(self._ids(seg)))
            else:
                parts.append(seg.to(self.device, torch.bfloat16))
        return torch.cat(parts)

    def chat(self, segments, prefix=''):
        return [self.chat_head] + list(segments) + [self.chat_tail + prefix]

    @torch.inference_mode()
    def generate(self, batch_segments, prefix='', max_new_tokens=64):
        embeds = [self.embed_segments(self.chat(s, prefix)) for s in batch_segments]
        # left padding for generation
        length = max(e.size(0) for e in embeds)
        pad = self.embed(torch.tensor([self.tokenizer.pad_token_id], device=self.device))
        inputs = torch.stack([torch.cat((pad.expand(length - e.size(0), -1), e)) for e in embeds])
        mask = torch.stack([
            torch.cat((torch.zeros(length - e.size(0), dtype=torch.long), torch.ones(e.size(0), dtype=torch.long)))
            for e in embeds
        ]).to(self.device)
        out = self.model.generate(
            inputs_embeds=inputs,
            attention_mask=mask,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id)
        return [self.tokenizer.decode(o, skip_special_tokens=True).strip() for o in out]

    @torch.inference_mode()
    def next_token_probs(self, batch_segments, candidates, prefix=''):
        # probabilities of single-token candidates at the next position, renormalized over the candidates
        embeds = [self.embed_segments(self.chat(s, prefix)) for s in batch_segments]
        length = max(e.size(0) for e in embeds)
        pad = self.embed(torch.tensor([self.tokenizer.pad_token_id], device=self.device))
        inputs = torch.stack([torch.cat((pad.expand(length - e.size(0), -1), e)) for e in embeds])
        mask = torch.stack([torch.arange(length, device=self.device) >= length - e.size(0) for e in embeds]).long()
        # left padding: positions of valid tokens start from 0
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        hidden = self.model.model(inputs_embeds=inputs, attention_mask=mask, position_ids=pos).last_hidden_state
        ids = []
        for c in candidates:
            tok = self.tokenizer(c, add_special_tokens=False).input_ids
            assert len(tok) == 1, f'candidate {c!r} is not a single token'
            ids.append(tok[0])
        ids = torch.tensor(ids, device=self.device)
        logits = self.model.lm_head(hidden[:, -1]).float()[:, ids]
        return logits.softmax(-1).cpu()

    def loss(self, batch_segments, targets, prefix=''):
        # cross entropy on the target strings only (teacher forcing), right padding
        embeds, labels = [], []
        for segs, tgt in zip(batch_segments, targets):
            prompt = self.embed_segments(self.chat(segs, prefix))
            tgt_ids = self._ids(tgt)
            embeds.append(torch.cat((prompt, self.embed(tgt_ids))))
            labels.append(torch.cat((torch.full((prompt.size(0), ), -100, device=self.device), tgt_ids)))
        length = max(e.size(0) for e in embeds)
        inputs = torch.stack([torch.cat((e, e.new_zeros(length - e.size(0), e.size(1)))) for e in embeds])
        mask = torch.stack([torch.arange(length, device=self.device) < e.size(0) for e in embeds]).long()
        labels = torch.stack([torch.cat((l, l.new_full((length - l.size(0), ), -100))) for l in labels])
        # only compute vocabulary logits where a label exists (the full [B, L, V] logits do not fit in memory)
        hidden = self.model.model(inputs_embeds=inputs, attention_mask=mask).last_hidden_state
        shift = labels[:, 1:]
        keep = shift != -100
        logits = self.model.lm_head(hidden[:, :-1][keep]).float()
        return torch.nn.functional.cross_entropy(logits, shift[keep])


# ===== prompt builders (same wording as videomind/eval/infer_caption.py) =====


def format_task(question, options):
    task = f'Question: {question}\nOptions:'
    for idx, opt in enumerate(options):
        task += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
    return task


def clip_lines(windows, contents, numbered):
    segs = []
    for i, ((s, e), c) in enumerate(zip(windows, contents)):
        head = f'Clip {i + 1} [{s:.1f}s - {e:.1f}s]: ' if numbered else f'[{s:.1f}s - {e:.1f}s] '
        segs += [head, c, '\n']
    if segs:
        segs[-1] = ''
    return segs


def score_prompt(duration, windows, contents, task, score_mode='options'):
    num = len(windows)
    tail = (f'\n\n{task}\n\n{SCORE_INSTRUCTIONS[score_mode]} Respond with JSON only, in the format '
            f'{{"scores": [s1, s2, ...]}}, with exactly {num} integers, one per clip in order.')
    return [f'You are given descriptions of {num} consecutive clips from a {duration:.1f}-second video.\n\n'
            ] + clip_lines(windows, contents, True) + [tail]


def answer_prompt(duration, coarse_windows, coarse_contents, fs, fe, fine_windows, fine_contents, task):
    segs = [f'You are given descriptions of a {duration:.1f}-second video.\n\nOverview of the whole video:\n'
            ] + clip_lines(coarse_windows, coarse_contents, False)
    if fine_windows:
        # fs=None: the fine clips come from several separate parts of the video
        head = 'the most relevant parts' if fs is None else f'the most relevant part ({fs:.1f}s - {fe:.1f}s)'
        segs += [f'\n\nDetailed descriptions of {head}:\n'] + clip_lines(fine_windows, fine_contents, False)
    return segs + [f'\n\n{task}\nPlease only give the best option.']


ANSWER_PREFIX = 'Best Option: ('


def clip_evidence_prompt(duration, window, content, task, num_options):
    # soft scoring of one clip: which option (if any) does this clip provide evidence for
    letters = ', '.join(chr(ord('A') + i) for i in range(num_options))
    return [f'Here is the description of one clip ({window[0]:.1f}s - {window[1]:.1f}s) of a {duration:.1f}-second '
            'video: ', content, f'\n\n{task}\n\nDoes this clip provide visual evidence for any of the options? '
            f'Answer with the option letter ({letters}), or N if it provides evidence for none of them.']


EVIDENCE_PREFIX = 'Answer: ('

# ===== multi-agent roles (same receiver LLM, different prompts) =====


def fmt_time(sec):
    sec = int(round(sec))
    return f'{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}' if sec >= 3600 else f'{sec // 60:02d}:{sec % 60:02d}'


def planner_prompt(duration, task):
    # the planner reads only the question (no video content) and plans what to look for
    return [f'You are the planner of a video question answering system. The video is {fmt_time(duration)} long '
            f'({duration:.0f} seconds). You cannot see the video. Plan how to find the answer.\n\n{task}\n\n'
            'Respond with JSON only, in the format {"scope": "local" or "global", "time_range": "mm:ss-mm:ss" or null, '
            '"targets": [...], "sub_questions": [...]}, where\n'
            '- scope: "global" if answering needs the whole video (e.g. an overall summary, the order of events across '
            'the video, or counting over the whole video), otherwise "local".\n'
            '- time_range: the time range of the video explicitly mentioned in the question, otherwise null.\n'
            '- targets: 1 to 3 short descriptions of what should be visible on screen at the moment that answers the '
            'question, written like captions of video clips (e.g. "a man washing dishes in a sink").\n'
            '- sub_questions: 1 to 3 simpler questions whose answers lead to the answer.']


def target_match_prompt(window, content, targets):
    # grounder with planner targets: does this clip show any of the targets
    listed = '\n'.join(f'- {t}' for t in targets)
    return [f'Here is the description of one clip ({window[0]:.1f}s - {window[1]:.1f}s) of a video: ', content,
            f'\n\nTargets:\n{listed}\n\nDoes this clip show any of the targets? Answer Yes or No.']


# candidates are ' Yes' / ' No' (single tokens with a leading space)
MATCH_PREFIX = 'Answer:'


def recon_prompt(content):
    return ['Here is an encoded description of a short video clip: ', content, '\nWrite out the description.']
