# Coarse-to-fine caption-based video QA (inspired by LLoVi, https://arxiv.org/abs/2312.17235)
#
# 1. coarse captioning: split the whole video into clips and caption each clip with a VLM
# 2. text grounding: an LLM scores the relevance of each coarse clip, and high-scoring neighbors are merged
# 3. fine captioning: densely caption around the top candidate, then score the fine clips to refine it
# 4. answering: answer from captions with the LLM (text), from a question-guided summary of the captions written by
#    the LLM (summary, following the sum_q / sum_qa prompts in https://github.com/CeeZh/LLoVi), or from the grounded
#    frames with the VLM (visual)
#
# The output format follows infer_auto.py so that eval_auto.py can be used directly.

import argparse
import copy
import hashlib
import json
import math
import os
import re

import nncore
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.utils import process_vision_info
from videomind.utils.io import get_duration
from videomind.utils.parser import parse_span

CAPTION_PROMPT = ('In one short sentence, describe the main action or change happening in this clip. '
                  'Do not describe clothing, colors, or the background.')

# bump this when CAPTION_PROMPT changes so that cached captions are not reused
CAPTION_VERSION = 'v2'

SCORE_PROMPT = """You are given descriptions of {num} consecutive clips from {scope}.

{timeline}

{task}

{instruction} Respond with JSON only, in the format {{"scores": [s1, s2, ...]}}, with exactly {num} integers, one per clip in order."""

# how clips are scored in grounding and refining (--score_mode)
SCORE_INSTRUCTIONS = {
    # rate generic relevance to the question
    'relevance':
    'Rate how relevant each clip is as visual evidence for this, from 0 (irrelevant) to 10 (shows the key evidence).',
    # first infer the target moment (see TARGET_PROMPT), then rate whether each clip belongs to it
    'moment':
    'The question asks about this moment: "{target}"\nRate how likely each clip belongs to this moment, from 0 '
    '(not this moment) to 10 (clearly this moment). Use temporal cues such as "after" or "before" to decide, instead of '
    'matching words in the question.',
    # rate whether each clip shows one of the answer options happening
    'options':
    'Rate how strongly each clip shows one of the answer options actually happening in the situation described by the '
    'question, from 0 (shows none of the options) to 10 (clearly shows one of the options).'
}

TARGET_PROMPT = """You are given descriptions of consecutive clips from a {duration:.1f}-second video.

{timeline}

{task}

Which moment of the video does this ask about? Answer in one sentence describing what happens on screen during that moment, and resolve temporal words such as "after", "before", "at the start", or "near the end" (e.g., "the moment right after the baby throws the green cup away, near the end of the video"). Do not answer the question itself."""

ANSWER_PROMPT = """You are given descriptions of a {duration:.1f}-second video.

Overview of the whole video:
{coarse}

Detailed descriptions of the most relevant part ({start:.1f}s - {end:.1f}s):
{fine}

{task}"""


# adapted from the 'sum_q' / 'sum_qa' and 'qa_sum' prompts in LLoVi (prompts.py)
SUMMARY_PROMPT = """You are given descriptions of a {duration:.1f}-second video. The descriptions are sequential, and each one describes a short clip.

Overview of the whole video:
{coarse}

Detailed descriptions of the most relevant part ({start:.1f}s - {end:.1f}s):
{fine}

Please give me a {num_words} words summary. When doing summarization, remember that your summary will be used to answer this multiple choice question: {question}{choices}"""

SUMMARY_CHOICES = """
Here are the choices.{options}
Do not answer this question directly. Instead, use the question and choices to guide your summary."""

SUMMARY_ANSWER_PROMPT = """You are given a summary of a {duration:.1f}-second video.

Summary: {summary}

{task}"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--pred_path')
    parser.add_argument('--vlm_path', default='model_zoo/Qwen2-VL-2B-Instruct')
    parser.add_argument('--llm_path', default='model_zoo/Qwen2.5-7B-Instruct')
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--answer_mode', default='text', choices=['text', 'summary', 'visual'])
    parser.add_argument('--summary_prompt', default='qa', choices=['q', 'qa'])
    parser.add_argument('--summary_words', type=int, default=100)
    parser.add_argument('--no_refine', action='store_true')
    parser.add_argument('--score_mode', default='relevance', choices=list(SCORE_INSTRUCTIONS))
    parser.add_argument('--coarse_sec', type=float, default=4.0)
    parser.add_argument('--max_coarse', type=int, default=32)
    parser.add_argument('--fine_sec', type=float, default=1.0)
    parser.add_argument('--max_fine', type=int, default=24)
    parser.add_argument('--caption_fps', type=float, default=2.0)
    parser.add_argument('--caption_pixels', type=int, default=128)
    parser.add_argument('--caption_batch', type=int, default=8)
    parser.add_argument('--merge_drop', type=float, default=2.0)
    parser.add_argument('--min_fine_len', type=float, default=2.0)
    parser.add_argument('--cache_dir', default='work_dirs/caption_cache')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()
    return args


def make_windows(s, e, sec, max_num):
    num = min(max(math.ceil((e - s) / sec), 1), max_num)
    step = (e - s) / num
    return [[round(s + i * step, 1), round(s + (i + 1) * step, 1)] for i in range(num)]


def format_timeline(windows, captions):
    return '\n'.join(f'[{s:.1f}s - {e:.1f}s] {c}' for (s, e), c in zip(windows, captions))


def format_numbered(windows, captions):
    return '\n'.join(f'Clip {i + 1} [{s:.1f}s - {e:.1f}s]: {c}' for i, ((s, e), c) in enumerate(zip(windows, captions)))


def format_task(anno):
    if 'question' in anno and 'options' in anno:
        task = f"Question: {anno['question']}\nOptions:"
        for idx, opt in enumerate(anno['options']):
            task += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
        return task
    return f"Query: {anno['query']}"


def parse_scores(response, num):
    match = re.search(r'"scores"\s*:\s*\[([^\]]*)\]', response) or re.search(r'\[([^\]]*)\]', response)
    if match is None:
        return None
    scores = [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?', match.group(1))]
    if len(scores) == 0:
        return None
    # tolerate slightly wrong lengths by truncating / padding with zeros
    return (scores + [0.0] * num)[:num]


def scores_to_segments(windows, scores, drop, max_num=3):
    # grow each peak (highest score first) into neighbors whose scores are within `drop` of the peak
    used, segs = set(), []
    for idx in sorted(range(len(scores)), key=lambda i: -scores[i]):
        if idx in used:
            continue
        peak, l, r = scores[idx], idx, idx
        while l - 1 >= 0 and l - 1 not in used and scores[l - 1] > 0 and scores[l - 1] >= peak - drop:
            l -= 1
        while r + 1 < len(scores) and r + 1 not in used and scores[r + 1] > 0 and scores[r + 1] >= peak - drop:
            r += 1
        used.update(range(l, r + 1))
        segs.append(([windows[l][0], windows[r][1]], peak / 10))
        if len(segs) == max_num:
            break
    return segs


def expand_span(seg, min_len, lo, hi):
    s, e = seg
    if e - s >= min_len:
        return seg
    min_len = min(min_len, hi - lo)
    c = (s + e) / 2
    # shift the window instead of truncating it when it hits the boundaries
    s = min(max(lo, c - min_len / 2), hi - min_len)
    return [round(s, 1), round(s + min_len, 1)]


class Captioner:

    def __init__(self, model_path, args):
        self.args = args
        self.processor = AutoProcessor.from_pretrained(model_path, do_resize=False)
        self.processor.tokenizer.padding_side = 'left'
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, attn_implementation='sdpa').cuda().eval()

        tag = f'{os.path.basename(os.path.normpath(model_path))}_{CAPTION_VERSION}_fps{args.caption_fps}_px{args.caption_pixels}'
        self.cache_dir = nncore.join(args.cache_dir, tag)
        nncore.mkdir(self.cache_dir)

    def _cache_path(self, video_path):
        return nncore.join(self.cache_dir, hashlib.md5(video_path.encode()).hexdigest() + '.json')

    def _load_cache(self, video_path):
        path = self._cache_path(video_path)
        try:
            return nncore.load(path) if nncore.is_file(path) else dict()
        except Exception:
            return dict()

    def _save_cache(self, video_path, cache):
        # atomic write since multiple chunks may share the same video
        path = self._cache_path(video_path)
        tmp = f'{path}.{os.getpid()}.tmp'
        with open(tmp, 'w') as f:
            json.dump(cache, f)
        os.replace(tmp, path)

    def build_messages(self, video_path, s, e):
        return [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'num_threads': self.args.num_threads,
                'video_start': s,
                'video_end': e,
                'min_pixels': 32 * 28 * 28,
                'max_pixels': self.args.caption_pixels * 28 * 28,
                'max_frames': 16,
                'fps': self.args.caption_fps
            }, {
                'type': 'text',
                'text': CAPTION_PROMPT
            }]
        }]

    @torch.inference_mode()
    def __call__(self, video_path, windows):
        cache = self._load_cache(video_path)
        keys = [f'{s:.1f}_{e:.1f}' for s, e in windows]
        todo = [i for i, k in enumerate(keys) if k not in cache]

        for b in range(0, len(todo), self.args.caption_batch):
            batch = todo[b:b + self.args.caption_batch]
            messages = [self.build_messages(video_path, *windows[i]) for i in batch]
            texts = [self.processor.apply_chat_template(m, add_generation_prompt=True) for m in messages]
            _, videos = process_vision_info(messages)
            data = self.processor(text=texts, videos=videos, padding=True, return_tensors='pt').to('cuda')

            output_ids = self.model.generate(**data, do_sample=False, max_new_tokens=40)
            output_ids = output_ids[:, data.input_ids.size(1):]
            responses = self.processor.batch_decode(output_ids, skip_special_tokens=True)

            for i, r in zip(batch, responses):
                cache[keys[i]] = r.strip().replace('\n', ' ')

        if len(todo) > 0:
            self._save_cache(video_path, cache)

        return [cache[k] for k in keys]

    @torch.inference_mode()
    def answer(self, video_path, s, e, prompt, dataset_cls):
        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'num_threads': self.args.num_threads,
                'video_start': s,
                'video_end': e,
                'min_pixels': getattr(dataset_cls, 'MIN_PIXELS', 128) * 28 * 28,
                'max_pixels': getattr(dataset_cls, 'MAX_PIXELS', 256) * 28 * 28,
                'max_frames': getattr(dataset_cls, 'MAX_FRAMES', 32),
                'fps': getattr(dataset_cls, 'FPS', 2.0)
            }, {
                'type': 'text',
                'text': prompt
            }]
        }]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True) + 'Best Option: ('
        images, videos = process_vision_info(messages)
        data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt').to('cuda')
        output_ids = self.model.generate(**data, do_sample=False, max_new_tokens=16)
        return self.processor.decode(output_ids[0, data.input_ids.size(1):], skip_special_tokens=True)


class Reasoner:

    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()

    @torch.inference_mode()
    def __call__(self, prompt, prefix='', max_new_tokens=128):
        messages = [{'role': 'user', 'content': prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefix
        data = self.tokenizer([text], return_tensors='pt').to('cuda')
        output_ids = self.model.generate(
            **data, do_sample=False, temperature=None, top_p=None, top_k=None, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(output_ids[0, data.input_ids.size(1):], skip_special_tokens=True).strip()


if __name__ == '__main__':
    args = parse_args()

    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'output_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'output.json')

    print(f'Dataset: {args.dataset}({args.split}) Chunk: {args.chunk} Index: {args.index} Output Path: {pred_path}')
    print(f'Answer mode: {args.answer_mode} Score mode: {args.score_mode} Refine: {not args.no_refine}')

    captioner = Captioner(args.vlm_path, args)
    reasoner = Reasoner(args.llm_path)

    dataset_cls = DATASETS.get(args.dataset)

    annos = dataset_cls.load_annos(split=args.split)
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    dumps = []
    for i in nncore.ProgressBar(range(len(annos))):
        anno = copy.deepcopy(annos[i])
        dump = copy.deepcopy(annos[i])

        video_path, duration, span = anno['video_path'], anno.get('duration'), anno.get('span')

        if duration is None:
            duration = get_duration(video_path, num_threads=args.num_threads)
            dump['duration'] = duration

        do_answering = all(k in anno for k in ('question', 'options'))
        task = format_task(anno)

        print()
        print(video_path, duration)
        print(task)

        dump['agents'] = ['coarse_captioner', 'text_grounder']

        # ===== 1. coarse captioning =====
        coarse_windows = make_windows(0, duration, args.coarse_sec, args.max_coarse)
        coarse_captions = captioner(video_path, coarse_windows)
        coarse_timeline = format_timeline(coarse_windows, coarse_captions)
        print(coarse_timeline)

        dump['coarse_windows'] = coarse_windows
        dump['coarse_captions'] = coarse_captions

        # ===== 2. text grounding by scoring coarse clips =====
        score_mode = args.score_mode if do_answering or args.score_mode != 'options' else 'relevance'

        if score_mode == 'moment':
            prompt = TARGET_PROMPT.format(duration=duration, timeline=coarse_timeline, task=task)
            target = reasoner(prompt, max_new_tokens=64).replace('\n', ' ')
            print(target)
            dump['target_moment'] = target
            instruction = SCORE_INSTRUCTIONS[score_mode].format(target=target.replace('"', "'"))
        else:
            instruction = SCORE_INSTRUCTIONS[score_mode]

        num = len(coarse_windows)
        prompt = SCORE_PROMPT.format(
            num=num,
            scope=f'a {duration:.1f}-second video',
            timeline=format_numbered(coarse_windows, coarse_captions),
            task=task,
            instruction=instruction)
        response = reasoner(prompt, max_new_tokens=4 * num + 32)
        print(response)

        scores = parse_scores(response, num)
        dump['ground_response'] = response
        dump['coarse_scores'] = scores
        dump['grounder_success'] = scores is not None

        if scores is None:
            print('WARNING: Failed to parse grounding response')
            cands, cand_conf = [[0, round(duration, 1)]], [0]
        else:
            segs = scores_to_segments(coarse_windows, scores, args.merge_drop)
            cands, cand_conf = [seg for seg, _ in segs], [c for _, c in segs]

        pred, conf = cands, cand_conf

        # ===== 3. fine captioning around the top candidate =====
        pad = duration / len(coarse_windows) / 2
        fs, fe = max(0, cands[0][0] - pad), min(duration, cands[0][1] + pad)
        fine_windows = make_windows(fs, fe, args.fine_sec, args.max_fine)
        fine_captions = captioner(video_path, fine_windows)
        fine_timeline = format_timeline(fine_windows, fine_captions)
        print(fine_timeline)

        dump['agents'].append('fine_captioner')
        dump['fine_windows'] = fine_windows
        dump['fine_captions'] = fine_captions

        if not args.no_refine:
            dump['agents'].append('text_refiner')

            num = len(fine_windows)
            prompt = SCORE_PROMPT.format(
                num=num,
                scope=f'the {fs:.1f}s - {fe:.1f}s part of a {duration:.1f}-second video',
                timeline=format_numbered(fine_windows, fine_captions),
                task=task,
                instruction=instruction)
            response = reasoner(prompt, max_new_tokens=4 * num + 32)
            print(response)

            fine_scores = parse_scores(response, num)
            dump['refine_response'] = response
            dump['fine_scores'] = fine_scores

            # refined segment always lies within the fine windows, fall back to coarse candidates otherwise
            if fine_scores is not None and max(fine_scores) > 0:
                (refined, _), = scores_to_segments(fine_windows, fine_scores, args.merge_drop, max_num=1)
                refined = expand_span(refined, args.min_fine_len, fs, fe)
                rest = [(c, cc) for c, cc in zip(cands, cand_conf) if c != refined]
                pred = [refined] + [c for c, _ in rest]
                conf = [1.0] + [cc for _, cc in rest]
                dump['pred_ori'] = cands
                dump['conf_ori'] = cand_conf
            elif fine_scores is None:
                print('WARNING: Failed to parse refine response')
            else:
                print('WARNING: No relevant fine clip, keeping coarse candidates')

        print(pred[0], span, duration)
        dump['pred'] = pred
        dump['conf'] = conf

        # ===== 4. answering =====
        if do_answering:
            dump['agents'].append(f'{args.answer_mode}_answerer')

            if args.answer_mode == 'text':
                prompt = ANSWER_PROMPT.format(
                    duration=duration, coarse=coarse_timeline, start=fs, end=fe, fine=fine_timeline, task=task)
                prompt += "\nPlease only give the best option."
                response = reasoner(prompt, prefix='Best Option: (', max_new_tokens=16)
            elif args.answer_mode == 'summary':
                choices = ''
                if args.summary_prompt == 'qa':
                    options = ''.join(f"\n({chr(ord('A') + k)}) {o[0].upper() + o[1:]}" for k, o in enumerate(anno['options']))
                    choices = SUMMARY_CHOICES.format(options=options)
                prompt = SUMMARY_PROMPT.format(
                    duration=duration,
                    coarse=coarse_timeline,
                    start=fs,
                    end=fe,
                    fine=fine_timeline,
                    num_words=args.summary_words,
                    question=anno['question'],
                    choices=choices)
                summary = reasoner(prompt, max_new_tokens=2 * args.summary_words + 32).replace('\n', ' ')
                print(summary)
                dump['summary'] = summary

                prompt = SUMMARY_ANSWER_PROMPT.format(duration=duration, summary=summary, task=task)
                prompt += "\nPlease only give the best option."
                response = reasoner(prompt, prefix='Best Option: (', max_new_tokens=16)
            else:
                # same moment expansion and prompt as the answerer in infer_auto.py
                if hasattr(dataset_cls, 'MIN_RATIO'):
                    min_len = duration * dataset_cls.MIN_RATIO
                else:
                    min_len = getattr(dataset_cls, 'MIN_LEN', 32)
                s, e = parse_span(pred[0], duration, min_len)
                prompt = task.replace('Question: ', '', 1) + '\nPlease only give the best option.'
                response = captioner.answer(video_path, s, e, prompt, dataset_cls)

            print(response, anno['ans'])
            dump['answerer_response'] = response
            dump['response'] = response

        dumps.append(dump)

    nncore.dump(dumps, pred_path)
