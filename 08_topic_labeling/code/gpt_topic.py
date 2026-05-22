"""gpt_topic.py

Generate short topic labels (summaries) for BERTopic topics using the OpenAI API.

Inputs
------
A topic-keywords CSV produced by your BERTopic keyword extraction step. Expected columns:
- topic
- count (optional)
- top_words  (comma-separated keywords)

Outputs
-------
Written into the output directory (default: /Users/grace/Desktop/new meta/gpt_topic):
- topic_summaries.csv         (topic, count, top_words, topic_summary)
- topic_summaries.jsonl       (one JSON object per line)

Notes
-----
- DO NOT hardcode API keys in code. Set OPENAI_API_KEY in your environment.
- This script uses the OpenAI Responses API (recommended).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
from openai import OpenAI


@dataclass
class Config:
    in_csv: Path
    out_dir: Path
    model: str
    top_k: int
    max_words: int
    temperature: float
    sleep_s: float
    dry_run: bool
    resume: bool

PROMPT_TEMPLATE = """You are assisting a researcher interpreting topic modeling results.

You will receive a list of keywords extracted from a statistical topic.

Your task is to propose a short descriptive label identifying the semantic domain represented by these keywords.

Guidelines:
- Do NOT repeat the keywords.
- Identify the main semantic domain suggested by the keywords.
- Prefer concise domain labels rather than abstract summaries.
- Use only information directly supported by the keywords.
- Do NOT add historical period labels, geographic labels, or cultural descriptors unless they explicitly appear in the keywords.
- Do NOT refer to specific authors.
- Avoid vague labels such as "human experience", "themes", "concepts", or "social roles" when a more concrete domain label is possible.
- Keep the label concise (max {max_words} words).
- Return only the label.

Keywords:
{keywords}

Topic label:"""

def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Generate GPT topic labels for BERTopic keywords")
    p.add_argument(
        "--in_csv",
        default="/Users/grace/Desktop/new meta/topic_modeling_keywords/topic_keywords_top10.csv",
        help="Path to topic keywords CSV (default: topic_keywords_top10.csv)",
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="Output directory",
    )
    p.add_argument(
        "--model",
        default="gpt-4.1-mini",
        help="OpenAI model slug (default: gpt-4.1-mini)",
    )
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--max_words", type=int, default=10)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--sleep_s", type=float, default=0.6)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")

    args = p.parse_args()

    in_csv = Path(args.in_csv).expanduser()
    if not in_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_csv}")

    default_out_dir = Path("/Users/grace/Desktop/new meta/gpt_topic")
    out_dir = Path(args.out_dir).expanduser() if args.out_dir else default_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    return Config(
        in_csv=in_csv,
        out_dir=out_dir,
        model=args.model,
        top_k=args.top_k,
        max_words=args.max_words,
        temperature=args.temperature,
        sleep_s=args.sleep_s,
        dry_run=bool(args.dry_run),
        resume=(not bool(args.no_resume)),
    )


def split_keywords(top_words: str, top_k: int) -> List[str]:
    parts = [p.strip() for p in str(top_words).split(",")]
    return [p for p in parts if p][:top_k]


def build_prompt(keywords: List[str], max_words: int) -> str:
    return PROMPT_TEMPLATE.format(keywords=", ".join(keywords), max_words=max_words)


def call_openai_label(client: OpenAI, model: str, prompt: str, temperature: float) -> str:
    resp = client.responses.create(
        model=model,
        input=prompt,
        temperature=temperature,
    )
    return (resp.output_text or "").strip()


def main() -> None:
    cfg = parse_args()

    print(f"🔎 Input: {cfg.in_csv}")
    print(f"📁 Output dir: {cfg.out_dir}")
    print(f"🤖 Model: {cfg.model}")

    df = pd.read_csv(cfg.in_csv)
    if not {"topic", "top_words"}.issubset(df.columns):
        raise ValueError("CSV must contain 'topic' and 'top_words' columns.")

    client = None if cfg.dry_run else OpenAI()

    jsonl_path = cfg.out_dir / "topic_summaries.jsonl"
    csv_path = cfg.out_dir / "topic_summaries.csv"

    seen = set()
    if cfg.resume and jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["topic"])
                except Exception:
                    pass

    mode = "a" if (cfg.resume and jsonl_path.exists()) else "w"
    results = []

    with open(jsonl_path, mode, encoding="utf-8") as jf:
        for _, row in df.iterrows():
            topic = row["topic"]
            if cfg.resume and topic in seen:
                continue

            keywords = split_keywords(row["top_words"], cfg.top_k)
            prompt = build_prompt(keywords, cfg.max_words)

            if cfg.dry_run:
                label = "DRY_RUN_LABEL"
            else:
                label = call_openai_label(client, cfg.model, prompt, cfg.temperature)
                time.sleep(cfg.sleep_s)

            rec = {
                "topic": topic,
                "count": row.get("count"),
                "top_words": ", ".join(keywords),
                "topic_summary": label,
            }

            jf.write(json.dumps(rec, ensure_ascii=False) + "\n")
            results.append(rec)
            print(f"topic={topic} -> {label}")

    if jsonl_path.exists():
        with open(jsonl_path, "r", encoding="utf-8") as f:
            results = [json.loads(line) for line in f if line.strip()]

    pd.DataFrame(results).to_csv(csv_path, index=False)

    print(f"\n✅ Saved: {csv_path}")
    print("🎉 Done.")


if __name__ == "__main__":
    main()
