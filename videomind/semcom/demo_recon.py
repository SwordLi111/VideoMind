# Show what the receiver LLM reconstructs from decoded soft tokens at different SNRs.
# Usage: python videomind/semcom/demo_recon.py --ckpt work_dirs/semcom/recon/recon_final.pth

import argparse
import random

import nncore
import torch

from videomind.semcom.model import build_semcom, split_soft
from videomind.semcom.receiver import Receiver, recon_prompt

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt')
parser.add_argument('--llm_path', default='model_zoo/Qwen2.5-7B-Instruct')
parser.add_argument('--num', type=int, default=8)
parser.add_argument('--snrs', type=float, nargs='+', default=[0, 5, 10, 20])
args = parser.parse_args()

torch.manual_seed(0)
receiver = Receiver(args.llm_path)
ckpt = torch.load(args.ckpt, map_location='cpu')
semcom, featurizer = build_semcom(ckpt['config'], receiver)
semcom.load_state_dict(ckpt['model'], strict=False)
semcom.eval()

annos = nncore.load('data/semcom/nextqa_causal_valid.json')
captions = random.Random(0).sample([c for a in annos[::10] for c in a['coarse_captions']], args.num)

feats, mask, _ = featurizer(captions)
for snr in args.snrs:
    with torch.no_grad():
        soft = split_soft(*semcom(feats, mask, snr))
    outs = receiver.generate([recon_prompt(s) for s in soft], max_new_tokens=40)
    # hard decision: nearest word embedding (cosine) for each decoded embedding
    vocab = torch.nn.functional.normalize(receiver.embed.weight.float(), dim=-1)
    hard = [
        receiver.tokenizer.decode((torch.nn.functional.normalize(s.float(), dim=-1) @ vocab.T).argmax(-1))
        for s in soft
    ] if soft[0].size(0) != ckpt['config'].get('num_tokens', -1) else None
    print(f'===== SNR {snr:g} dB =====')
    for i, (c, o) in enumerate(zip(captions, outs)):
        print(f'  sent: {c}\n  soft: {o}' + (f'\n  hard: {hard[i]}' if hard else ''))
