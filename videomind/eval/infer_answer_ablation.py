# Ablate the grounded moment given to the VideoMind answerer on NExT-GQA causal questions.
#
# The answerer is the same as in infer_auto.py (VideoMind-2B with all adapters disabled, i.e. Qwen2-VL-2B, same
# prompt and video sampling). Only the moment it watches changes:
#   reuse:  the moment predicted by VideoMind (grounder + verifier), read from a previous infer_auto.py run (--ref)
#   none:   the whole video
#   random: one random ~4s clip (same clip partition as videomind/eval/infer_caption.py)
#   anti:   a random ~4s clip that does not overlap the ground-truth evidence
#   oracle: the range covering all ground-truth evidence spans
#   other:  a random ~4s clip of a different random video (the question is unchanged)
#
# --min_len 32 follows infer_auto.py (the moment is expanded to at least 32s), --min_len 0 keeps it as is.

import argparse
import copy
import random

import nncore
import torch

from videomind.dataset.hybrid import DATASETS
from videomind.dataset.utils import process_vision_info
from videomind.eval.infer_caption import make_windows
from videomind.model.builder import build_model
from videomind.utils.io import get_duration
from videomind.utils.parser import parse_span


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path')
    parser.add_argument('--model_path', default='model_zoo/VideoMind-2B')
    parser.add_argument('--ground_mode', default='reuse', choices=['reuse', 'none', 'random', 'anti', 'oracle', 'other'])
    parser.add_argument('--ref', default='outputs_semcom/videomind_causal_test/output.json')
    parser.add_argument('--min_len', type=float, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    pred_path = nncore.join(args.pred_path, f'output_{args.index}.json' if args.chunk > 1 else 'output.json')
    print(f'Ground mode: {args.ground_mode} Min len: {args.min_len} Output: {pred_path}')

    dataset_cls = DATASETS.get('nextgqa')
    annos = [a for a in dataset_cls.load_annos(split='test') if a['task'].startswith('C')]
    all_videos = sorted({a['video_path']: a['duration'] for a in annos}.items())
    ref = {(d['video_path'], d['question']): d['pred'][0] for d in nncore.load(args.ref)}
    annos = [annos[i::args.chunk] for i in range(args.chunk)][args.index]

    model, processor = build_model(args.model_path)
    device = next(model.parameters()).device

    dumps = []
    for anno in nncore.ProgressBar(annos):
        dump = copy.deepcopy(anno)
        video_path, duration = anno['video_path'], anno['duration']
        rng = random.Random(f"{args.seed}-{video_path}-{anno['question']}")
        clips = make_windows(0, duration, 4.0, 32)

        if args.ground_mode == 'reuse':
            seg = ref[(video_path, anno['question'])]
        elif args.ground_mode == 'none':
            seg = [0, duration]
        elif args.ground_mode == 'random':
            seg = rng.choice(clips)
        elif args.ground_mode == 'anti':

            def overlap(w):
                return sum(max(0, min(w[1], e) - max(w[0], s)) for s, e in anno['span'])

            least = min(overlap(w) for w in clips)
            seg = rng.choice([w for w in clips if overlap(w) == least])
        elif args.ground_mode == 'oracle':
            seg = [min(s for s, _ in anno['span']), max(e for _, e in anno['span'])]
        else:
            video_path, duration = rng.choice([v for v in all_videos if v[0] != anno['video_path']])
            duration = duration or get_duration(video_path)
            seg = rng.choice(make_windows(0, duration, 4.0, 32))
            dump['other_video'] = video_path

        s, e = parse_span(seg, duration, args.min_len if args.min_len > 0 else -1)
        dump['pred'], dump['conf'], dump['answer_window'] = [list(seg)], [1.0], [s, e]

        prompt = anno['question'] + '\nOptions:'
        for idx, opt in enumerate(anno['options']):
            prompt += f"\n({chr(ord('A') + idx)}) {opt[0].upper() + opt[1:]}"
        prompt += '\nPlease only give the best option.'

        # same answerer input as infer_auto.py
        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video_path,
                'num_threads': 1,
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
        text = processor.apply_chat_template(messages, add_generation_prompt=True) + 'Best Option: ('
        images, videos = process_vision_info(messages)
        data = processor(text=[text], images=images, videos=videos, return_tensors='pt').to(device)

        with model.disable_adapter():
            output_ids = model.generate(
                **data, do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=None,
                max_new_tokens=256)
        output_ids = output_ids[0, data.input_ids.size(1):]
        if output_ids[-1] == processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        dump['response'] = processor.decode(output_ids, clean_up_tokenization_spaces=False)
        dumps.append(dump)

    nncore.dump(dumps, pred_path)


if __name__ == '__main__':
    main()
