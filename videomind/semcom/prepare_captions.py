# Pre-compute question-agnostic captions for the semantic communication experiments on NExT-QA causal questions.
#
# For every video, two sets of captions are generated (and stored in the same cache as infer_caption.py):
#   coarse: ~4s clips covering the whole video (sent in round 1)
#   fine:   1s clips on an integer-aligned grid covering the whole video, so that any segment requested by the
#           receiver after round 1 can be served without running the captioner again (sent in round 2)
#
# Usage (one shard): python videomind/semcom/prepare_captions.py --split train --chunk 8 --index 0
# Then build the index: python videomind/semcom/prepare_captions.py --split train --build_index

import argparse
import csv
import glob
import math
import os
from types import SimpleNamespace

import nncore

from videomind.eval.infer_caption import CAPTION_VERSION, Captioner, make_windows
from videomind.utils.io import get_duration

ANNO_PATHS = {'train': 'data/nextgqa/train.csv', 'valid': 'data/nextgqa/val.csv', 'test': 'data/nextgqa/test.csv'}
SPAN_PATHS = {'valid': 'data/nextgqa/gsub_val.json', 'test': 'data/nextgqa/gsub_test.json'}
VIDEO_ID_MAP = 'data/nextgqa/map_vid_vidorID.json'
VIDEO_ROOT = 'data/nextqa/videos'
INDEX_PATH = 'data/semcom/nextqa_causal_{}.json'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train', choices=['train', 'valid', 'test'])
    parser.add_argument('--vlm_path', default='model_zoo/Qwen2-VL-2B-Instruct')
    parser.add_argument('--coarse_sec', type=float, default=4.0)
    parser.add_argument('--max_coarse', type=int, default=32)
    parser.add_argument('--caption_fps', type=float, default=2.0)
    parser.add_argument('--caption_pixels', type=int, default=128)
    parser.add_argument('--caption_batch', type=int, default=8)
    parser.add_argument('--cache_dir', default='work_dirs/caption_cache')
    parser.add_argument('--num_threads', type=int, default=1)
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--build_index', action='store_true')
    parser.add_argument('--allow_missing', action='store_true', help='skip videos whose captions are not ready yet')
    return parser.parse_args()


def fine_grid(duration):
    # integer-aligned 1s clips, the last one is clipped to the video end
    return [[float(k), round(min(k + 1.0, duration), 1)] for k in range(math.ceil(duration)) if k < duration - 0.05]


def load_causal(split):
    with open(ANNO_PATHS[split]) as f:
        rows = [r for r in csv.DictReader(f) if r['type'].startswith('C')]
    video_map = nncore.load(VIDEO_ID_MAP)
    spans = nncore.load(SPAN_PATHS[split]) if split in SPAN_PATHS else None

    annos = []
    for r in rows:
        options = [r[f'a{k}'][0].upper() + r[f'a{k}'][1:] for k in range(5)]
        answer = r['answer'][0].upper() + r['answer'][1:]
        anno = dict(
            qid=f"{r['video_id']}_{r['qid']}",
            video_id=r['video_id'],
            video_path=nncore.join(VIDEO_ROOT, video_map[r['video_id']] + '.mp4'),
            question=r['question'][0].upper() + r['question'][1:] + '?',
            options=options,
            ans=chr(ord('A') + options.index(answer)),
            task=r['type'])
        if spans is not None:
            # test / valid only: ground truth evidence and duration from NExT-GQA
            if r['video_id'] not in spans or r['qid'] not in spans[r['video_id']]['location']:
                continue
            anno['span'] = spans[r['video_id']]['location'][r['qid']]
            anno['duration'] = spans[r['video_id']]['duration']
        annos.append(anno)
    return annos


def main():
    args = parse_args()
    annos = load_causal(args.split)

    videos = dict()
    for a in annos:
        videos.setdefault(a['video_id'], a)
    vids = sorted(videos)
    print(f'Split: {args.split} Causal questions: {len(annos)} Videos: {len(vids)}')

    captioner_args = SimpleNamespace(
        caption_fps=args.caption_fps,
        caption_pixels=args.caption_pixels,
        caption_batch=args.caption_batch,
        cache_dir=args.cache_dir,
        num_threads=args.num_threads)

    if args.build_index:
        # read everything back from the cache (no GPU needed when all captions exist)
        captioner = Captioner.__new__(Captioner)
        captioner.args = captioner_args
        # must match the cache tag in Captioner.__init__
        name = os.path.basename(os.path.normpath(args.vlm_path))
        tag = f'{name}_{CAPTION_VERSION}_fps{args.caption_fps}_px{args.caption_pixels}'
        captioner.cache_dir = nncore.join(args.cache_dir, tag)

        durations = dict()
        for path in glob.glob(f'data/semcom/durations_{args.split}_*.json'):
            durations.update(nncore.load(path))
        index, skipped = [], set()
        for a in annos:
            vid = a['video_id']
            if 'duration' not in a and vid not in durations:
                # the shard of this video has not finished yet
                assert args.allow_missing, f'{vid}: duration missing, run captioning first'
                skipped.add(vid)
                continue
            duration = a.get('duration') or durations[vid]
            cache = captioner._load_cache(a['video_path'])
            coarse = make_windows(0, duration, args.coarse_sec, args.max_coarse)
            fine = fine_grid(duration)
            keys = [f'{s:.1f}_{e:.1f}' for s, e in coarse + fine]
            missing = [k for k in keys if k not in cache]
            if len(missing) > 0:
                assert args.allow_missing, f'{vid}: {len(missing)} captions missing, run captioning first'
                skipped.add(vid)
                continue
            a = dict(a, duration=duration)
            a['coarse_windows'] = coarse
            a['coarse_captions'] = [cache[f'{s:.1f}_{e:.1f}'] for s, e in coarse]
            a['fine_windows'] = fine
            a['fine_captions'] = [cache[f'{s:.1f}_{e:.1f}'] for s, e in fine]
            index.append(a)
        nncore.dump(index, INDEX_PATH.format(args.split))
        print(f'Saved {len(index)} samples to {INDEX_PATH.format(args.split)} (skipped {len(skipped)} videos)')
        return

    captioner = Captioner(args.vlm_path, captioner_args)

    vids = [vids[i::args.chunk] for i in range(args.chunk)][args.index]
    durations = dict()
    for vid in nncore.ProgressBar(vids):
        a = videos[vid]
        duration = a.get('duration') or get_duration(a['video_path'], num_threads=args.num_threads)
        durations[vid] = duration
        captioner(a['video_path'], make_windows(0, duration, args.coarse_sec, args.max_coarse) + fine_grid(duration))

    nncore.dump(durations, f'data/semcom/durations_{args.split}_{args.index}.json')


if __name__ == '__main__':
    main()
