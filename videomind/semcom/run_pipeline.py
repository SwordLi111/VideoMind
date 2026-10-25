# Two-round question answering over a (semantic) communication channel on NExT-QA causal questions (--data
# nextqa_causal) or LVBench (--data lvbench), using the indexes built by prepare_captions.py / prepare_lvbench.py.
#
#   round 1: transmitter sends the coarse captions of the whole video -> receiver LLM scores the clips with the
#            question and options, and picks a segment
#   feedback: receiver sends back only the requested time span (assumed error free)
#   round 2: transmitter sends the 1s-grid fine captions inside the span -> receiver LLM answers
#
# --ground_mode llm:    the receiver LLM scores the coarse clips (default)
#               agent:  multi-agent receiver, step 1 (planner): the planner reads only the question and plans the
#                       search (scope, explicit time range, visual targets, sub-questions); questions with a time
#                       range request that range directly, global questions spread the fine budget over the whole
#                       video, other questions are grounded by scoring each coarse clip against the planned targets
#               random: request one uniformly random coarse clip (no LLM scoring)
#               oracle: request the ground-truth evidence span (upper bound of grounding)
#               none:   no round 2, answer from the coarse captions only
#               blind:  no captions at all, answer from the question and options only (language prior)
#               anti:   request a random coarse clip that does not overlap the ground-truth evidence (wrong place)
#               other:  like random, but the fine captions are taken from another video (wrong content)
#
# --score_mode soft:  score each coarse clip separately by the log-odds log((1 - P(N)) / P(N)) when the receiver LLM is
#                     asked which option the clip provides evidence for (A/B/C/D or N for none); continuous scores that
#                     do not saturate near 1, no output parsing
# --select merge:  request the fine clips around the top contiguous segment (default)
#          topk:   request fine clips from the top --topk coarse clips, splitting a budget of --fine_budget fine clips
#                  in proportion to softmax(score / --soft_temp)
#
# --channel text:   captions reach the receiver as plain text (noiseless upper bound)
# --channel semcom: captions go through semantic encoder -> AWGN (--snr) -> semantic decoder (--ckpt)
#
# The output follows infer_auto.py, so videomind/eval/eval_auto.py can be used for evaluation.

import argparse
import copy
import json
import math
import random
import re

import nncore
import torch

from videomind.eval.infer_caption import parse_scores, scores_to_segments
from videomind.semcom.model import build_semcom, split_soft
from videomind.semcom.receiver import (ANSWER_PREFIX, EVIDENCE_PREFIX, MATCH_PREFIX, Receiver, answer_prompt,
                                       clip_evidence_prompt, format_task, planner_prompt, score_prompt,
                                       target_match_prompt)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='nextqa_causal', choices=['nextqa_causal', 'lvbench'])
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'])
    parser.add_argument('--pred_path')
    parser.add_argument('--llm_path', default='model_zoo/Qwen2.5-7B-Instruct')
    parser.add_argument('--channel', default='text', choices=['text', 'semcom'])
    parser.add_argument('--ckpt')
    parser.add_argument('--snr', type=float, default=10.0)
    parser.add_argument('--ground_mode', default='llm', choices=['llm', 'agent', 'random', 'oracle', 'none', 'blind', 'anti', 'other'])
    parser.add_argument('--score_mode', default='options')
    parser.add_argument('--merge_drop', type=float, default=2.0)
    parser.add_argument('--max_fine', type=int, default=24)
    parser.add_argument('--select', default='merge', choices=['merge', 'topk'])
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--fine_budget', type=int, default=30)
    parser.add_argument('--soft_temp', type=float, default=1.0, help='softmax temperature on the log-odds scores')
    parser.add_argument('--score_batch', type=int, default=32)
    parser.add_argument('--no_answer', action='store_true')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    return parser.parse_args()


def select_fine(anno, seg, num_coarse, max_fine):
    # request the fine clips around the top segment (same padding as infer_caption.py)
    duration = anno['duration']
    pad = duration / num_coarse / 2
    fs, fe = max(0, seg[0] - pad), min(duration, seg[1] + pad)
    idx = [i for i, (s, e) in enumerate(anno['fine_windows']) if s < fe and e > fs]
    if len(idx) > max_fine:
        start = (len(idx) - max_fine) // 2
        idx = idx[start:start + max_fine]
    return fs, fe, idx


def select_topk(anno, scores, k, budget, temp):
    # split the fine budget over the top-k coarse clips in proportion to softmax(score / temp)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    w = torch.tensor([scores[i] for i in order]).div(temp).softmax(0).tolist()
    clips = [[j for j, (s, e) in enumerate(anno['fine_windows']) if s < anno['coarse_windows'][i][1] and
              e > anno['coarse_windows'][i][0]] for i in order]
    ideal = [budget * x for x in w]
    n = [min(len(c), max(1, int(x))) for c, x in zip(clips, ideal)]
    while sum(n) < budget and any(m < len(c) for m, c in zip(n, clips)):
        i = max((i for i in range(len(n)) if n[i] < len(clips[i])), key=lambda i: ideal[i] - n[i])
        n[i] += 1
    idx = set()
    for c, m in zip(clips, n):
        # evenly spaced fine clips inside each selected coarse clip
        idx.update(c[round(t * (len(c) - 1) / max(1, m - 1))] for t in range(m)) if m > 1 else idx.add(c[len(c) // 2])
    return order, n, sorted(idx)


TIME = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b')


def parse_time(t):
    sec = 0
    for p in t.strip().split(':'):
        sec = sec * 60 + float(p)
    return sec


def parse_plan(text, question, duration):
    plan = dict(scope='local', time_range=None, targets=[], sub_questions=[], raw=text)
    match = re.search(r'\{.*\}', text, flags=re.S)
    try:
        out = json.loads(match.group(0)) if match else dict()
    except json.JSONDecodeError:
        out = dict()
    plan['parsed'] = bool(out)
    if out.get('scope') == 'global':
        plan['scope'] = 'global'
    plan['targets'] = [str(t) for t in (out.get('targets') or [])][:3]
    plan['sub_questions'] = [str(q) for q in (out.get('sub_questions') or [])][:3]
    # only trust a time range when the question itself contains a timestamp
    tr = out.get('time_range')
    if isinstance(tr, str) and TIME.search(question):
        try:
            s, e = [parse_time(x) for x in tr.split('-')]
            if 0 <= s < e and s < duration:
                plan['time_range'] = [s, min(e, duration)]
        except ValueError:
            pass
    return plan


def select_range(anno, span, budget):
    # fine clips inside an explicit time range: evenly subsample if too many, extend both sides if too few
    fw = anno['fine_windows']
    idx = [i for i, (s, e) in enumerate(fw) if s < span[1] and e > span[0]]
    if not idx:
        c = min(range(len(fw)), key=lambda i: abs((fw[i][0] + fw[i][1]) / 2 - span[0]))
        idx = [c]
    if len(idx) > budget:
        idx = [idx[round(t * (len(idx) - 1) / max(1, budget - 1))] for t in range(budget)]
    lo, hi = idx[0], idx[-1]
    while len(idx) < budget and (lo > 0 or hi < len(fw) - 1):
        if lo > 0:
            lo -= 1
            idx.insert(0, lo)
        if len(idx) < budget and hi < len(fw) - 1:
            hi += 1
            idx.append(hi)
    return sorted(set(idx))


def select_uniform(anno, budget):
    n = len(anno['fine_windows'])
    return sorted({round(t * (n - 1) / max(1, budget - 1)) for t in range(min(budget, n))})


class Channel:

    def __init__(self, args, receiver):
        self.args, self.receiver = args, receiver
        if args.channel == 'semcom':
            ckpt = torch.load(args.ckpt, map_location='cpu')
            self.semcom, self.featurizer = build_semcom(ckpt['config'], receiver)
            self.semcom.load_state_dict(ckpt['model'], strict=False)
            self.semcom.eval()

    @torch.no_grad()
    def __call__(self, captions):
        if self.args.channel == 'text':
            return list(captions)
        feats, mask, _ = self.featurizer(captions)
        return split_soft(*self.semcom(feats, mask, self.args.snr))


def main():
    args = parse_args()
    torch.manual_seed(args.seed + args.index)

    if args.chunk > 1:
        pred_path = nncore.join(args.pred_path, f'output_{args.index}.json')
    else:
        pred_path = nncore.join(args.pred_path, 'output.json')
    print(f'Split: {args.split} Channel: {args.channel} SNR: {args.snr} Ground: {args.ground_mode} Output: {pred_path}')

    receiver = Receiver(args.llm_path)
    channel = Channel(args, receiver)

    annos = nncore.load(f'data/semcom/{args.data}_{args.split}.json')
    # fine captions of all videos of the split, used by --ground_mode other
    all_fine = {a['video_id']: a['fine_captions'] for a in annos}
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    dumps = []
    for b in nncore.ProgressBar(range(0, len(annos), args.batch_size)):
        batch = [copy.deepcopy(a) for a in annos[b:b + args.batch_size]]
        tasks = [format_task(a['question'], a['options']) for a in batch]

        # ===== round 1: coarse captions of the whole video =====
        # the receiver keeps what it got in round 1 and reuses it when answering
        received = [channel(a['coarse_captions']) for a in batch]

        plans = [None] * len(batch)
        if args.ground_mode == 'agent':
            # planner: reads only the question and plans the search
            outs = receiver.generate([planner_prompt(a['duration'], t) for a, t in zip(batch, tasks)], max_new_tokens=256)
            plans = [parse_plan(o, a['question'], a['duration']) for o, a in zip(outs, batch)]

        # which questions are grounded by soft scoring, and against what
        kinds = [None] * len(batch)
        for k in range(len(batch)):
            if args.ground_mode == 'llm' and args.score_mode == 'soft':
                kinds[k] = 'option'
            elif args.ground_mode == 'agent' and plans[k]['time_range'] is None and plans[k]['scope'] == 'local':
                kinds[k] = 'target' if plans[k]['targets'] else 'option'

        def batched_probs(items, make_prompt, candidates, prefix):
            probs = []
            for t in range(0, len(items), args.score_batch):
                sub = items[t:t + args.score_batch]
                probs += receiver.next_token_probs([make_prompt(k, j) for k, j in sub], candidates, prefix).tolist()
            return probs

        soft = [None] * len(batch)
        items = [(k, j) for k in range(len(batch)) if kinds[k] == 'option' for j in range(len(batch[k]['coarse_windows']))]
        if items:
            # one forward pass per (question, coarse clip), read P(A..D, N) at the next token
            letters = [chr(ord('A') + i) for i in range(len(batch[items[0][0]]['options']))] + ['N']
            probs = batched_probs(
                items, lambda k, j: clip_evidence_prompt(batch[k]['duration'], batch[k]['coarse_windows'][j], received[k][
                    j], tasks[k], len(batch[k]['options'])), letters, EVIDENCE_PREFIX)
            for (k, j), p in zip(items, probs):
                soft[k] = (soft[k] or []) + [p]
        items = [(k, j) for k in range(len(batch)) if kinds[k] == 'target' for j in range(len(batch[k]['coarse_windows']))]
        if items:
            # grounder with the planned targets: P(Yes) / P(No) that the clip shows any target
            probs = batched_probs(
                items, lambda k, j: target_match_prompt(batch[k]['coarse_windows'][j], received[k][j], plans[k][
                    'targets']), [' Yes', ' No'], MATCH_PREFIX)
            for (k, j), p in zip(items, probs):
                soft[k] = (soft[k] or []) + [p]

        if args.ground_mode == 'llm' and args.score_mode != 'soft':
            prompts = [
                score_prompt(a['duration'], a['coarse_windows'], coarse, task, args.score_mode)
                for a, coarse, task in zip(batch, received, tasks)
            ]
            responses = receiver.generate(
                prompts, max_new_tokens=4 * max(len(a['coarse_windows']) for a in batch) + 32)

        # ===== feedback: requested span =====
        for k, a in enumerate(batch):
            if args.ground_mode == 'agent':
                a['plan'] = plans[k]
            if kinds[k] is not None:
                if kinds[k] == 'option':
                    # evidence score = log-odds of "some option" vs "none", full precision (1 - P(N) saturates near 1)
                    a['option_probs'] = soft[k]
                    scores = [math.log(max(1 - p[-1], 1e-12)) - math.log(max(p[-1], 1e-12)) for p in soft[k]]
                else:
                    # target score = log-odds of "Yes" vs "No"
                    scores = [math.log(max(p[0], 1e-12)) - math.log(max(p[1], 1e-12)) for p in soft[k]]
                a['coarse_scores'], a['grounder_success'], a['score_kind'] = scores, True, kinds[k]
                # merge neighbors with positive log-odds within --merge_drop of the peak
                segs = scores_to_segments(a['coarse_windows'], scores, args.merge_drop)
            elif args.ground_mode == 'agent' and plans[k]['time_range'] is not None:
                segs = [(plans[k]['time_range'], 1)]
            elif args.ground_mode == 'agent':
                # global scope
                segs = [([0, round(a['duration'], 1)], 0)]
            elif args.ground_mode == 'llm':
                r = responses[k]
                scores = parse_scores(r, len(a['coarse_windows']))
                a['ground_response'], a['coarse_scores'], a['grounder_success'] = r, scores, scores is not None
                if scores is None:
                    segs = [([0, round(a['duration'], 1)], 0)]
                else:
                    segs = scores_to_segments(a['coarse_windows'], scores, args.merge_drop)
            elif args.ground_mode == 'anti':
                rng = random.Random(f"{args.seed}-{a['qid']}")

                def overlap(w):
                    return sum(max(0, min(w[1], e) - max(w[0], s)) for s, e in a['span'])

                least = min(overlap(w) for w in a['coarse_windows'])
                segs = [(rng.choice([w for w in a['coarse_windows'] if overlap(w) == least]), 0)]
            elif args.ground_mode in ('random', 'other'):
                # reproducible per question
                rng = random.Random(f"{args.seed}-{a['qid']}")
                segs = [(a['coarse_windows'][rng.randrange(len(a['coarse_windows']))], 0)]
            elif args.ground_mode == 'oracle' and a.get('span'):
                # range covering all ground-truth evidence spans
                segs = [([min(s for s, _ in a['span']), max(e for _, e in a['span'])], 1)]
            else:
                segs = [([0, round(a['duration'], 1)], 0)]
            a['pred'], a['conf'] = [s for s, _ in segs], [c for _, c in segs]
            if args.ground_mode == 'agent' and kinds[k] is None:
                if plans[k]['time_range'] is not None:
                    a['fine_idx'] = select_range(a, plans[k]['time_range'], args.fine_budget)
                    a['fs'], a['fe'] = a['fine_windows'][a['fine_idx'][0]][0], a['fine_windows'][a['fine_idx'][-1]][1]
                else:
                    a['fine_idx'] = select_uniform(a, args.fine_budget)
                    a['fs'] = a['fe'] = None
            elif args.ground_mode in ('llm', 'agent') and args.select == 'topk' and a.get('coarse_scores'):
                order, counts, a['fine_idx'] = select_topk(a, a['coarse_scores'], args.topk, args.fine_budget,
                                                           args.soft_temp)
                # several separate parts: no single span header in the answer prompt
                a['fs'] = a['fe'] = None
                a['pred'] = [a['coarse_windows'][i] for i in order]
                a['conf'] = [a['coarse_scores'][i] for i in order]
                a['selected_clips'], a['fine_counts'] = order, counts
            elif args.ground_mode in ('none', 'blind'):
                a['fs'], a['fe'], a['fine_idx'] = 0, a['duration'], []
            else:
                a['fs'], a['fe'], a['fine_idx'] = select_fine(a, a['pred'][0], len(a['coarse_windows']),
                                                              args.max_fine)

        # ===== round 2: fine captions inside the requested span, then answer =====
        if not args.no_answer:
            prompts = []
            for a, coarse, task in zip(batch, received, tasks):
                if args.ground_mode == 'blind':
                    prompts.append([f'{task}\nPlease only give the best option.'])
                    continue
                fine_caps = [a['fine_captions'][i] for i in a['fine_idx']]
                if args.ground_mode == 'other' and fine_caps:
                    # same number of consecutive fine captions from a different random video
                    rng = random.Random(f"{args.seed}-{a['qid']}-other")
                    pool = all_fine[rng.choice([v for v in all_fine if v != a['video_id']])]
                    start = rng.randrange(max(1, len(pool) - len(fine_caps) + 1))
                    fine_caps = (pool[start:] + pool * len(fine_caps))[:len(fine_caps)]
                    a['other_fine'] = fine_caps
                fine = channel(fine_caps) if fine_caps else []
                prompts.append(
                    answer_prompt(a['duration'], a['coarse_windows'], coarse, a['fs'], a['fe'],
                                  [a['fine_windows'][i] for i in a['fine_idx']], fine, task))
            responses = receiver.generate(prompts, prefix=ANSWER_PREFIX, max_new_tokens=8)
            for a, r in zip(batch, responses):
                a['response'] = r

        for a in batch:
            a['snr'] = args.snr if args.channel == 'semcom' else None
            # keep dumps small
            for k in ('coarse_captions', 'fine_captions', 'fine_windows'):
                a.pop(k)
            dumps.append(a)

    nncore.dump(dumps, pred_path)


if __name__ == '__main__':
    main()
