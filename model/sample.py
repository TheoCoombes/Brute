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

from data import decode_compact, encode_compact, sample_ban_ids, sentence_end_ids
from model import HaemmrConfig, HaemmrLM


def build_codec(compact_to_gpt2: torch.Tensor):
    """Return (encode, decode) closures backed by the GPT-2 tokenizer."""
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode(text: str) -> torch.Tensor:
        return encode_compact(text, tok, compact_to_gpt2)

    def decode(ids: torch.Tensor) -> str:
        return decode_compact(ids, tok, compact_to_gpt2)

    return encode, decode, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="./haemmr.pt")
    p.add_argument("--prompt", default="The history of")
    p.add_argument("--n", type=int, default=60, help="number of tokens to generate")
    p.add_argument("--sentence", action="store_true",
                   help="stop after sentence punctuation once --min-new tokens are generated")
    p.add_argument("--min-new", type=int, default=12)
    p.add_argument("--clean", action="store_true",
                   help="ban OOV, control, non-ASCII, and continuation-fragment tokens")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20, help="0 = full Boltzmann sampling")
    p.add_argument("--rep-window", type=int, default=3)
    p.add_argument("--ban-oov", action="store_true", default=True)
    p.add_argument("--device", default=None, choices=[None, "cpu", "mps", "cuda"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device("cpu")

    blob = torch.load(args.ckpt, weights_only=False, map_location="cpu")
    cfg = HaemmrConfig(**blob["model"]["cfg"])
    model = HaemmrLM(cfg, device=device)
    model.load_state_dict(blob["model"])
    compact_to_gpt2 = blob["compact_to_gpt2"]
    encode, decode, tok = build_codec(compact_to_gpt2)

    ids = encode(args.prompt).unsqueeze(0).to(device)
    ban = sample_ban_ids(compact_to_gpt2, tok, clean=args.clean) if args.ban_oov or args.clean else None
    stop = sentence_end_ids(compact_to_gpt2, tok) if args.sentence else None
    out = model.generate(ids, args.n, temperature=args.temperature, top_k=args.top_k,
                         ban_ids=ban, repetition_window=args.rep_window,
                         stop_ids=stop, min_new=args.min_new)
    print(decode(out[0]).strip())


if __name__ == "__main__":
    main()
