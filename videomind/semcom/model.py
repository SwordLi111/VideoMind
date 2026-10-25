# Trainable semantic encoder / decoder with an AWGN channel in between. Two architectures (--arch):
#
# pool (SemCom):        caption -> S complex symbols for the whole caption -> K soft tokens
# token (TokenSemCom):  DeepSC-style, every caption token -> n complex symbols -> one decoded embedding that is
#                       supervised directly against the receiver LLM's word embedding of that token
#
#   caption -> Featurizer (frozen) -> semantic encoder -> complex symbols (unit average power)
#           -> AWGN channel -> semantic decoder -> embeddings read by the receiver LLM
#
# Featurizer modes (--tx_features):
#   embed: word embeddings of the receiver LLM (no context)
#   vlm:   last hidden states of the language model inside the transmitter's captioner (Qwen2-VL-2B)
#
# Complex symbols are stored as real tensors of shape [..., S, 2] (real, imag).

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration


def power_normalize(x):
    # unit average power per complex symbol for each sample
    power = x.pow(2).sum(dim=(1, 2), keepdim=True) / x.size(1)
    return x / power.clamp_min(1e-8).sqrt()


def awgn(x, snr_db):
    # x: [B, S, 2] with unit average power per complex symbol, snr_db: float or tensor of shape [B]
    snr_db = torch.as_tensor(snr_db, dtype=x.dtype, device=x.device).reshape(-1, 1, 1)
    sigma = torch.sqrt(10**(-snr_db / 10) / 2)
    return x + sigma * torch.randn_like(x)


class SemanticEncoder(nn.Module):

    def __init__(self, llm_dim, d_model=512, num_layers=4, num_heads=8, num_symbols=16, max_len=64):
        super().__init__()
        self.in_proj = nn.Linear(llm_dim, d_model)
        self.pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, 4 * d_model, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        # one learnable query per output symbol, attending over the caption tokens
        self.queries = nn.Parameter(torch.randn(1, num_symbols, d_model) * 0.02)
        self.pool = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, 2)

    def forward(self, tok_embeds, mask):
        # tok_embeds: [B, L, llm_dim] frozen word embeddings, mask: [B, L] with True for valid tokens
        h = self.in_proj(tok_embeds.float()) + self.pos[:, :tok_embeds.size(1)]
        h = self.encoder(h, src_key_padding_mask=~mask)
        z, _ = self.pool(self.queries.expand(h.size(0), -1, -1), h, h, key_padding_mask=~mask)
        return power_normalize(self.out(self.norm(z)))


class SemanticDecoder(nn.Module):

    def __init__(self,
                 llm_dim,
                 d_model=512,
                 num_layers=4,
                 num_heads=8,
                 num_symbols=16,
                 num_tokens=8,
                 embed_rms=0.01,
                 out_norm='ln'):
        super().__init__()
        self.out_norm = out_norm
        self.in_proj = nn.Linear(2, d_model)
        # the receiver knows the SNR (e.g. from channel estimation) and can adapt to it
        self.snr_proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, num_symbols + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, 4 * d_model, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.queries = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        layer = nn.TransformerDecoderLayer(d_model, num_heads, 4 * d_model, 0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, 2)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, llm_dim)
        if out_norm == 'ln':
            # soft tokens are layer-normalized then scaled to the RMS of the LLM word embeddings
            self.scale = nn.Parameter(torch.tensor(float(embed_rms)))
        else:
            # plain linear output, initialized so that the output RMS roughly matches the LLM word embeddings
            nn.init.normal_(self.out.weight, std=embed_rms / d_model**0.5)
            nn.init.zeros_(self.out.bias)

    def forward(self, y, snr_db):
        # y: [B, S, 2] received symbols, snr_db: float or tensor of shape [B]
        snr_db = torch.as_tensor(snr_db, dtype=y.dtype, device=y.device).reshape(-1, 1, 1).expand(y.size(0), 1, 1)
        h = torch.cat((self.snr_proj(snr_db / 20), self.in_proj(y)), dim=1) + self.pos
        h = self.encoder(h)
        t = self.decoder(self.queries.expand(h.size(0), -1, -1), h)
        e = self.out(self.norm(t))
        if self.out_norm == 'ln':
            e = F.layer_norm(e, e.shape[-1:]) * self.scale
        return e


class SemCom(nn.Module):
    """Transmitter (semantic encoder) + AWGN channel + receiver (semantic decoder)."""

    def __init__(self,
                 llm_dim,
                 embed_rms,
                 d_model=512,
                 num_layers=4,
                 num_symbols=16,
                 num_tokens=8,
                 max_len=64,
                 enc_in_dim=None,
                 out_norm='ln',
                 tx_features='embed'):
        super().__init__()
        self.max_len, self.tx_features = max_len, tx_features
        self.encoder = SemanticEncoder(
            enc_in_dim or llm_dim, d_model, num_layers, num_symbols=num_symbols, max_len=max_len)
        self.decoder = SemanticDecoder(
            llm_dim,
            d_model,
            num_layers,
            num_symbols=num_symbols,
            num_tokens=num_tokens,
            embed_rms=embed_rms,
            out_norm=out_norm)

    def forward(self, feats, mask, snr_db):
        # feats: [B, L, C] from Featurizer, mask: [B, L], snr_db=None means a noiseless channel
        # returns soft tokens [B, K, D] and their validity mask [B, K]
        x = self.encoder(feats, mask)
        y = x if snr_db is None else awgn(x, snr_db)
        e = self.decoder(y, 100.0 if snr_db is None else snr_db)
        return e, torch.ones(e.shape[:2], dtype=torch.bool, device=e.device)


class TokenSemCom(nn.Module):
    """DeepSC-style token-level transmission: each caption token -> n complex symbols -> one decoded embedding."""

    def __init__(self,
                 llm_dim,
                 embed_rms,
                 d_model=512,
                 num_layers=4,
                 num_heads=8,
                 sym_per_token=4,
                 max_len=64,
                 enc_in_dim=None,
                 **kwargs):
        super().__init__()
        self.max_len, self.sym_per_token = max_len, sym_per_token
        self.enc_in = nn.Linear(enc_in_dim or llm_dim, d_model)
        self.enc_pos = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, 4 * d_model, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.enc_norm = nn.LayerNorm(d_model)
        self.enc_out = nn.Linear(d_model, 2 * sym_per_token)

        self.dec_in = nn.Linear(2 * sym_per_token, d_model)
        self.snr_proj = nn.Linear(1, d_model)
        self.dec_pos = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, 4 * d_model, 0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.dec_norm = nn.LayerNorm(d_model)
        self.dec_out = nn.Linear(d_model, llm_dim)
        nn.init.normal_(self.dec_out.weight, std=embed_rms / d_model**0.5)
        nn.init.zeros_(self.dec_out.bias)

    def encode(self, feats, mask):
        b, l = mask.shape
        h = self.enc_in(feats.float()) + self.enc_pos[:, :l]
        h = self.encoder(h, src_key_padding_mask=~mask)
        x = self.enc_out(self.enc_norm(h)).view(b, l, self.sym_per_token, 2) * mask[..., None, None]
        # unit average power per transmitted complex symbol, padding positions are not transmitted
        power = x.pow(2).sum(dim=(1, 2, 3)) / (mask.sum(1) * self.sym_per_token)
        return x / power.clamp_min(1e-8).sqrt()[:, None, None, None]

    def decode(self, y, mask, snr_db):
        b, l = mask.shape
        snr_db = torch.as_tensor(snr_db, dtype=y.dtype, device=y.device).reshape(-1, 1, 1).expand(b, 1, 1)
        h = torch.cat((self.snr_proj(snr_db / 20), self.dec_in(y.flatten(2))), dim=1) + self.dec_pos[:, :l + 1]
        key_mask = torch.cat((mask.new_ones(b, 1), mask), dim=1)
        h = self.decoder(h, src_key_padding_mask=~key_mask)
        return self.dec_out(self.dec_norm(h[:, 1:]))

    def forward(self, feats, mask, snr_db):
        # returns decoded embeddings [B, L, D] (one per caption token) and the token mask [B, L]
        x = self.encode(feats, mask)
        y = x if snr_db is None else awgn(x.flatten(1, 2), snr_db).view_as(x)
        return self.decode(y, mask, 100.0 if snr_db is None else snr_db), mask


class Featurizer:
    """Frozen caption features at the transmitter."""

    def __init__(self, mode, receiver, vlm_path='model_zoo/Qwen2-VL-2B-Instruct', max_len=64):
        self.mode, self.max_len, self.device = mode, max_len, receiver.device
        if mode == 'embed':
            self.tokenizer, self.embed = receiver.tokenizer, receiver.embed
            self.dim = receiver.embed.weight.size(1)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(vlm_path)
        # right padding so that positions of valid tokens start from 0
        self.tokenizer.padding_side = 'right'
        if mode != 'embed':
            vlm = Qwen2VLForConditionalGeneration.from_pretrained(vlm_path, torch_dtype=torch.bfloat16)
            # only the language model is needed to encode caption text
            self.lm = vlm.model.to(self.device).eval().requires_grad_(False)
            self.dim = vlm.config.hidden_size
            del vlm.visual

    @torch.no_grad()
    def __call__(self, captions):
        tok = self.tokenizer(
            list(captions),
            add_special_tokens=False,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt').to(self.device)
        mask = tok.attention_mask.bool()
        if self.mode == 'embed':
            feats = self.embed(tok.input_ids)
        else:
            feats = self.lm(input_ids=tok.input_ids, attention_mask=tok.attention_mask).last_hidden_state
        return feats.float(), mask, tok.input_ids


def build_semcom(config, receiver):
    """Build SemCom + Featurizer from a checkpoint config (older checkpoints lack the newer keys)."""
    config = dict(config)
    tx_features = config.get('tx_features', 'embed')
    featurizer = Featurizer(tx_features, receiver, max_len=config.get('max_len', 64))
    weight = receiver.embed.weight
    cls = TokenSemCom if config.pop('arch', 'pool') == 'token' else SemCom
    semcom = cls(
        weight.size(1),
        weight.float().pow(2).mean().sqrt().item(),
        enc_in_dim=featurizer.dim,
        **{k: v for k, v in config.items() if k != 'enc_in_dim'})
    return semcom.to(receiver.device), featurizer


def split_soft(soft, mask):
    """[B, K, D] soft tokens + [B, K] mask -> list of per-caption tensors with only the valid tokens."""
    return [e[m] for e, m in zip(soft, mask)]
