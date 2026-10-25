# Train the semantic encoder / decoder (videomind/semcom/model.py) with a frozen receiver LLM.
#
# --stage recon: learn to transmit captions
#   --arch pool:  the receiver LLM must reproduce each caption from its decoded soft tokens
#   --arch token: every decoded embedding must identify its caption token among the LLM vocabulary (cosine-similarity
#                 cross entropy) and match its word embedding (MSE), optionally plus the LLM reproduction loss
# --stage task:  answer NExT-QA causal questions from the two-round transmission (coarse captions of the whole video +
#                fine captions of the span requested after noiseless round-1 grounding), plus the recon losses above
#
# The SNR is sampled uniformly from [--snr_min, --snr_max] for every caption (recon) or question (task).

import argparse
import math
import random
import time

import nncore
import torch
import torch.nn.functional as F

from videomind.semcom.model import build_semcom, split_soft
from videomind.semcom.receiver import ANSWER_PREFIX, Receiver, answer_prompt, format_task, recon_prompt

EOS = '<|im_end|>'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='recon', choices=['recon', 'task'])
    parser.add_argument('--work_dir', default='work_dirs/semcom/debug')
    parser.add_argument('--init', help='checkpoint to start from (e.g. the recon stage)')
    parser.add_argument('--llm_path', default='model_zoo/Qwen2.5-7B-Instruct')
    parser.add_argument('--train_grounding', default='outputs_semcom/text_train', help='round-1 grounding of train')
    parser.add_argument('--valid_grounding', default='outputs_semcom/text_valid', help='round-1 grounding of valid')
    parser.add_argument('--arch', default='pool', choices=['pool', 'token'])
    parser.add_argument('--sym_per_token', type=int, default=4, help='complex symbols per caption token (token arch)')
    parser.add_argument('--mse_weight', type=float, default=1.0, help='token arch: weight of the embedding MSE')
    parser.add_argument('--llm_recon_weight', type=float, default=0.0, help='token arch, recon stage')
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--num_symbols', type=int, default=16)
    parser.add_argument('--num_tokens', type=int, default=8)
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--tx_features', default='embed', choices=['embed', 'vlm'])
    parser.add_argument('--out_norm', default='ln', choices=['ln', 'none'])
    parser.add_argument('--snr_min', type=float, default=0.0)
    parser.add_argument('--snr_max', type=float, default=20.0)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--recon_weight', type=float, default=0.5)
    parser.add_argument('--recon_per_batch', type=int, default=16)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    return parser.parse_args()


def load_split(split, grounding_dir):
    annos = nncore.load(f'data/semcom/nextqa_causal_{split}.json')
    if grounding_dir is None or not nncore.is_dir(grounding_dir):
        return annos, None
    grounded = dict()
    for path in nncore.ls(grounding_dir, ext='json', join_path=True):
        for d in nncore.load(path):
            grounded[d['qid']] = d
    samples = []
    for a in annos:
        g = grounded.get(a['qid'])
        if g is not None:
            samples.append(dict(a, fs=g['fs'], fe=g['fe'], fine_idx=g['fine_idx']))
    return annos, samples


class Trainer:

    def __init__(self, args):
        self.args = args
        self.receiver = Receiver(args.llm_path, gradient_checkpointing=args.gradient_checkpointing)
        if args.arch == 'pool':
            self.config = dict(
                d_model=args.d_model,
                num_layers=args.num_layers,
                num_symbols=args.num_symbols,
                num_tokens=args.num_tokens,
                max_len=args.max_len,
                tx_features=args.tx_features,
                out_norm=args.out_norm)
        else:
            # decoded embeddings are supervised against the receiver's own word embeddings of the same tokens
            assert args.tx_features == 'embed', 'token arch needs the receiver tokenizer at the transmitter'
            self.config = dict(
                arch='token',
                d_model=args.d_model,
                num_layers=args.num_layers,
                sym_per_token=args.sym_per_token,
                max_len=args.max_len,
                tx_features='embed')
        self.model, self.featurizer = build_semcom(self.config, self.receiver)
        weight = self.receiver.embed.weight.detach()
        self.vocab = F.normalize(weight.float(), dim=-1).to(torch.bfloat16)
        self.embed_rms = weight.float().pow(2).mean().sqrt().item()
        if args.init:
            ckpt = torch.load(args.init, map_location='cpu')
            assert ckpt['config'] == self.config, (ckpt['config'], self.config)
            self.model.load_state_dict(ckpt['model'], strict=False)
            print(f'Loaded {args.init}')

    def transmit(self, captions, snrs):
        feats, mask, ids = self.featurizer(captions)
        snr = torch.tensor(snrs, device='cuda', dtype=torch.float)
        soft, out_mask = self.model(feats, mask, snr)
        return split_soft(soft, out_mask), (soft, out_mask, ids)

    def token_losses(self, raw):
        # token arch only: identify each token among the vocabulary + match its word embedding
        soft, mask, ids = raw
        e, t = soft[mask], ids[mask]
        logits = (F.normalize(e.float(), dim=-1).to(torch.bfloat16) @ self.vocab.T).float() * 20
        ce = F.cross_entropy(logits, t)
        mse = (e - self.receiver.embed(t).float()).pow(2).mean() / self.embed_rms**2
        acc = (logits.argmax(-1) == t).float().mean()
        return ce, mse, acc

    def llm_recon_loss(self, soft, captions):
        return self.receiver.loss([recon_prompt(s) for s in soft], [c + EOS for c in captions])

    def recon_loss(self, captions, snrs):
        # returns the total loss and a dict of its parts
        soft, raw = self.transmit(captions, snrs)
        if self.args.arch == 'pool':
            loss = self.llm_recon_loss(soft, captions)
            return loss, dict(recon=loss.item())
        ce, mse, acc = self.token_losses(raw)
        loss = ce + self.args.mse_weight * mse
        parts = dict(ce=ce.item(), mse=mse.item(), acc=acc.item())
        if self.args.llm_recon_weight > 0:
            recon = self.llm_recon_loss(soft, captions)
            loss = loss + self.args.llm_recon_weight * recon
            parts['recon'] = recon.item()
        return loss, parts

    def task_loss(self, samples, snrs):
        # transmit all captions of the batch in one pass, each question with its own SNR
        captions, caption_snrs, sizes = [], [], []
        for s, snr in zip(samples, snrs):
            caps = s['coarse_captions'] + [s['fine_captions'][i] for i in s['fine_idx']]
            captions += caps
            caption_snrs += [snr] * len(caps)
            sizes.append(len(caps))
        soft, raw = self.transmit(captions, caption_snrs)

        prompts, offset = [], 0
        for s, n in zip(samples, sizes):
            st = soft[offset:offset + n]
            offset += n
            nc = len(s['coarse_captions'])
            prompts.append(
                answer_prompt(s['duration'], s['coarse_windows'], list(st[:nc]), s['fs'], s['fe'],
                              [s['fine_windows'][i] for i in s['fine_idx']], list(st[nc:]),
                              format_task(s['question'], s['options'])))
        loss = self.receiver.loss(prompts, [s['ans'] for s in samples], prefix=ANSWER_PREFIX)

        # recon losses keep the soft tokens interpretable
        if self.args.arch == 'pool':
            # on a random subset of the transmitted captions
            k = min(self.args.recon_per_batch, len(captions))
            idx = random.sample(range(len(captions)), k)
            recon = self.llm_recon_loss([soft[i] for i in idx], [captions[i] for i in idx])
        else:
            ce, mse, _ = self.token_losses(raw)
            recon = ce + self.args.mse_weight * mse
        return loss, recon

    @torch.no_grad()
    def evaluate(self, valid_captions, valid_samples):
        self.model.eval()
        out = dict()
        for snr in (0.0, 10.0, 20.0):
            losses, accs = [], []
            for b in range(0, len(valid_captions), 32):
                caps = valid_captions[b:b + 32]
                # LLM reproduction loss, comparable across architectures
                soft, raw = self.transmit(caps, [snr] * len(caps))
                losses.append(self.llm_recon_loss(soft, caps).item())
                if self.args.arch == 'token':
                    accs.append(self.token_losses(raw)[2].item())
            out[f'recon@{snr:g}dB'] = sum(losses) / len(losses)
            if accs:
                out[f'tok_acc@{snr:g}dB'] = sum(accs) / len(accs)
            if valid_samples:
                losses, correct = [], 0
                for b in range(0, len(valid_samples), 4):
                    batch = valid_samples[b:b + 4]
                    losses.append(self.task_loss(batch, [snr] * len(batch))[0].item())
                out[f'task@{snr:g}dB'] = sum(losses) / len(losses)
        self.model.train()
        return out

    def save(self, name):
        state = self.model.state_dict()
        path = nncore.join(self.args.work_dir, name)
        torch.save(dict(model=state, config=self.config, args=vars(self.args)), path)
        print(f'Saved {path}')


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    nncore.mkdir(args.work_dir)

    trainer = Trainer(args)

    train_annos, train_samples = load_split('train', args.train_grounding if args.stage == 'task' else None)
    valid_annos, valid_samples = load_split('valid', args.valid_grounding if args.stage == 'task' else None)

    # unique captions of all train videos (coarse + fine), each video counted once
    seen, train_captions = set(), []
    for a in train_annos:
        if a['video_id'] not in seen:
            seen.add(a['video_id'])
            train_captions += a['coarse_captions'] + a['fine_captions']
    rng = random.Random(0)
    valid_captions = rng.sample([c for a in valid_annos[::10] for c in a['coarse_captions']], 256)
    if valid_samples:
        valid_samples = rng.sample(valid_samples, min(128, len(valid_samples)))
    print(f'Stage: {args.stage} Train captions: {len(train_captions)} '
          f'Train questions: {len(train_samples or [])} Valid questions: {len(valid_samples or [])}')
    if args.stage == 'task':
        assert train_samples, f'run round-1 grounding on train first ({args.train_grounding})'

    params = [p for p in trainer.model.parameters() if p.requires_grad]
    print(f'Trainable parameters: {sum(p.numel() for p in params) / 1e6:.1f}M')
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1))))

    trainer.model.train()
    t0, logs = time.time(), dict()
    for step in range(1, args.steps + 1):
        if args.stage == 'recon':
            caps = random.sample(train_captions, args.batch_size)
            snrs = [random.uniform(args.snr_min, args.snr_max) for _ in caps]
            loss, parts = trainer.recon_loss(caps, snrs)
        else:
            batch = random.sample(train_samples, args.batch_size)
            snrs = [random.uniform(args.snr_min, args.snr_max) for _ in batch]
            task, recon = trainer.task_loss(batch, snrs)
            loss = task + args.recon_weight * recon
            parts = dict(task=task.item(), recon=recon.item())

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        sched.step()

        for k, v in parts.items():
            logs.setdefault(k, []).append(v)
        if step % args.log_every == 0:
            msg = ' '.join(f'{k}: {sum(v) / len(v):.4f}' for k, v in logs.items())
            print(f'[{step}/{args.steps}] {msg} lr: {sched.get_last_lr()[0]:.2e} '
                  f'time: {(time.time() - t0) / step:.2f}s/it', flush=True)
            logs = dict()
        if step % args.eval_every == 0 or step == args.steps:
            print(f'[{step}/{args.steps}] valid:', {k: round(v, 4) for k, v in trainer.evaluate(valid_captions, valid_samples).items()},
                  flush=True)
            trainer.save('latest.pth')

    trainer.save(f'{args.stage}_final.pth')


if __name__ == '__main__':
    main()
