"""Sample text from a trained HÆMMR checkpoint (min-Hamming / Boltzmann decode).

    python sample.py --ckpt haemmr.pt --prompt "The history of" --n 60 --temperature 0.8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import brute  # noqa: F401

from model import HaemmrConfig, HaemmrLM


def build_codec(compact_to_gpt2: torch.Tensor):
    """Return (encode, decode) closures backed by the GPT-2 tokenizer."""
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    gpt2_to_compact = torch.full((int(compact_to_gpt2.max().item()) + 1,), -1,
                                 dtype=torch.long)
    gpt2_to_compact[compact_to_gpt2] = torch.arange(compact_to_gpt2.numel(),
                                                    dtype=torch.long)

    def encode(text: str) -> torch.Tensor:
        gpt2_ids = tok.encode(text)
        if not gpt2_ids:
            eos = int(getattr(tok, "eos_token_id", 0) or 0)
            return torch.tensor([eos], dtype=torch.long)
        gi = torch.tensor(gpt2_ids, dtype=torch.long)
        if int(gi.max().item()) >= gpt2_to_compact.numel():
            raise ValueError("checkpoint vocabulary does not cover the prompt token ids")
        ids = gpt2_to_compact[gi]
        if (ids < 0).any():
            raise ValueError("prompt contains tokens outside the checkpoint vocabulary")
        return ids

    def decode(ids: torch.Tensor) -> str:
        return tok.decode(compact_to_gpt2[ids.long().cpu()].tolist())

    return encode, decode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="./haemmr.pt")
    p.add_argument("--prompt", default="The history of")
    p.add_argument("--n", type=int, default=60, help="number of tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20, help="0 = full Boltzmann sampling")
    p.add_argument("--rep-window", type=int, default=3)
    p.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cpu")

    blob = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    cfg = HaemmrConfig(**blob["model"]["cfg"])
    model = HaemmrLM(cfg, device=device)
    model.load_state_dict(blob["model"])
    encode, decode = build_codec(blob["compact_to_gpt2"])

    ids = encode(args.prompt).unsqueeze(0).to(device)
    out = model.generate(ids, args.n, temperature=args.temperature, top_k=args.top_k,
                         repetition_window=args.rep_window)
    print(decode(out[0]))


if __name__ == "__main__":
    main()
