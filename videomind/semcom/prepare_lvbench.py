# Pre-compute question-agnostic captions of LVBench for the semantic communication experiments.
#
#   coarse: --coarse_sec clips (default 60s) covering the whole video, sent in round 1
#   fine:   --fine_sec clips (default 4s) on a fixed grid covering the whole video, so that any span requested by the
#           receiver after round 1 can be served without running the captioner again (round 2)
#
# Captions are stored in the same cache as videomind/eval/infer_caption.py.
#
# Usage (one shard): python videomind/semcom/prepare_lvbench.py --chunk 12 --index 0
# Then build the index: python videomind/semcom/prepare_lvbench.py --build_index

import argparse
import glob
import math
import os
from types import SimpleNamespace

import nncore

from videomind.dataset.hybrid import DATASETS
from videomind.eval.infer_caption import CAPTION_VERSION, Captioner, make_windows
from videomind.utils.io import get_duration

INDEX_PATH = 'data/semcom/lvbench_test.json'
DURATION_PATH = 'data/semcom/durations_lvbench_{}.json'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm_path', default='model_zoo/Qwen2-VL-2B-Instruct')
    parser.add_argument('--coarse_sec', type=float, default=60.0)
    parser.add_argument('--fine_sec', type=float, default=4.0)
    parser.add_argument('--caption_fps', type=float, default=2.0)
    parser.add_argument('--caption_pixels', type=int, default=128)
    parser.add_argument('--caption_batch', type=int, default=8)
    parser.add_argument('--cache_dir', default='work_dirs/caption_cache')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--build_index', action='store_true')
    return parser.parse_args()


def coarse_windows(duration, sec):
    # no cap on the number of clips for hour-long videos
    return make_windows(0, duration, sec, 10**6)


def fine_grid(duration, sec):
    # fixed grid aligned to multiples of `sec`, the last clip is clipped to the video end
    return [[round(k * sec, 1), round(min((k + 1) * sec, duration), 1)] for k in range(math.ceil(duration / sec))
            if k * sec < duration - 0.05]


def parse_time(t):
    # "mm:ss" or "hh:mm:ss" -> seconds
    sec = 0
    for p in t.strip().split(':'):
        sec = sec * 60 + float(p)
    return sec


def parse_span(ref):
    # a few references are incomplete (e.g. "08:17-None"), these questions are kept for QA but have no span
    try:
        s, e = ref.split('-')
        return [[parse_time(s), parse_time(e)]]
    except ValueError:
        return None


def main():
    args = parse_args()
    annos = DATASETS.get('lvbench').load_annos(split='test')
    vids = sorted({a['video_path'] for a in annos})
    print(f'LVBench questions: {len(annos)} Videos: {len(vids)}')

    captioner_args = SimpleNamespace(
        caption_fps=args.caption_fps,
        caption_pixels=args.caption_pixels,
        caption_batch=args.caption_batch,
        cache_dir=args.cache_dir,
        num_threads=1)

    if args.build_index:
        captioner = Captioner.__new__(Captioner)
        captioner.args = captioner_args
        # must match the cache tag in Captioner.__init__
        name = os.path.basename(os.path.normpath(args.vlm_path))
        captioner.cache_dir = nncore.join(args.cache_dir, f'{name}_{CAPTION_VERSION}_fps{args.caption_fps}_px{args.caption_pixels}')

        durations = dict()
        for path in glob.glob(DURATION_PATH.format('*')):
            durations.update(nncore.load(path))

        caches, index = dict(), []
        for i, a in enumerate(annos):
            path, duration = a['video_path'], durations[a['video_path']]
            if path not in caches:
                cache = captioner._load_cache(path)
                coarse, fine = coarse_windows(duration, args.coarse_sec), fine_grid(duration, args.fine_sec)
                missing = [w for w in coarse + fine if f'{w[0]:.1f}_{w[1]:.1f}' not in cache]
                assert len(missing) == 0, f'{path}: {len(missing)} captions missing, run captioning first'
                caches[path] = dict(
                    coarse_windows=coarse,
                    coarse_captions=[cache[f'{s:.1f}_{e:.1f}'] for s, e in coarse],
                    fine_windows=fine,
                    fine_captions=[cache[f'{s:.1f}_{e:.1f}'] for s, e in fine])
            span = parse_span(a['time_reference'])
            index.append(
                dict(
                    qid=f'lvbench_{i}',
                    video_id=os.path.basename(path)[:-4],
                    video_path=path,
                    duration=duration,
                    question=a['question'],
                    options=a['options'],
                    ans=a['ans'],
                    task=a['task'],
                    **({} if span is None else dict(span=span)),
                    **caches[path]))
        nncore.dump(index, INDEX_PATH)
        print(f'Saved {len(index)} samples to {INDEX_PATH}')
        return

    captioner = Captioner(args.vlm_path, captioner_args)
    durations = dict()
    for path in nncore.ProgressBar([vids[i::args.chunk] for i in range(args.chunk)][args.index]):
        duration = get_duration(path)
        durations[path] = duration
        captioner(path, coarse_windows(duration, args.coarse_sec) + fine_grid(duration, args.fine_sec))
    nncore.dump(durations, DURATION_PATH.format(args.index))


if __name__ == '__main__':
    main()
