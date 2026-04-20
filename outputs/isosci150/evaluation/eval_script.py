#!/usr/bin/env python3
"""
IsoSci-150 Standalone Evaluation Script
=========================================
Evaluates any model via OpenRouter on the IsoSci-150 test set.

Usage:
    python eval_script.py --model openai/gpt-4o --split test --api-key YOUR_KEY
    
    # Or with environment variable:
    export OPENROUTER_API_KEY=your_key
    python eval_script.py --model openai/gpt-4o
"""

import argparse, json, re, time
from pathlib import Path
import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PROMPT_TEMPLATE = (
    "Think step by step. Show your reasoning clearly. "
    "Provide your final answer at the end in the format: "
    "**Final Answer:** <your answer>\n\n{question}"
)

def call_model(api_key, model, question, max_tokens=2048, temperature=0.0):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://isosci-benchmark.github.io",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(3):
        try:
            r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if r.status_code == 429:
                time.sleep(10 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if attempt == 2: raise
            time.sleep(5)

def extract_answer(text):
    m = re.search(r'\*\*Final Answer[:\*]*\*?\s*(.+?)(?:\n|$)', text, re.I)
    if m: return m.group(1).strip()
    nums = re.findall(r'-?[\d]+\.?[\d]*(?:[eE][+-]?\d+)?', text)
    return nums[-1] if nums else text.strip().split("\n")[-1]

def is_correct(pred, gold):
    def to_num(s):
        nums = re.findall(r'-?[\d]+\.?[\d]*(?:[eE][+-]?\d+)?', re.sub(r'[^\d.\-e+]', ' ', s))
        try: return float(nums[0]) if nums else None
        except: return None
    p, g = to_num(pred), to_num(gold)
    if p is not None and g is not None and g != 0:
        return abs(p - g) / abs(g) <= 0.02
    return pred.strip().upper()[:1] == gold.strip().upper()[:1]

def evaluate(model, api_key, split="test"):
    data_path = Path(__file__).parent.parent / f"{split}.json"
    with open(data_path) as f: pairs = json.load(f)
    
    results = {"model": model, "split": split, "pairs": []}
    src_correct = tgt_correct = 0
    
    for i, pair in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] {pair['mapping']}", end="", flush=True)
        
        src_resp = call_model(api_key, model, pair["source"]["question"])
        src_pred = extract_answer(src_resp)
        src_ok = is_correct(src_pred, pair["source"]["answer"])
        
        tgt_resp = call_model(api_key, model, pair["target"]["question"])
        tgt_pred = extract_answer(tgt_resp)
        tgt_ok = is_correct(tgt_pred, pair["target"]["answer"])
        
        src_correct += src_ok; tgt_correct += tgt_ok
        results["pairs"].append({
            "pair_id": pair["pair_id"],
            "source_correct": src_ok,
            "target_correct": tgt_ok,
            "source_pred": src_pred,
            "target_pred": tgt_pred,
        })
        print(f" src={'✓' if src_ok else '✗'} tgt={'✓' if tgt_ok else '✗'}")
        time.sleep(0.3)
    
    n = len(pairs)
    results["source_accuracy"] = round(src_correct / n * 100, 1)
    results["target_accuracy"] = round(tgt_correct / n * 100, 1)
    results["domain_gap_delta"] = round((src_correct - tgt_correct) / n * 100, 1)
    
    out = f"results_{model.replace('/', '_')}_{split}.json"
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSource acc: {results['source_accuracy']}%")
    print(f"Target acc: {results['target_accuracy']}%")
    print(f"Domain gap δ: {results['domain_gap_delta']}pp")
    print(f"Results saved to {out}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    import os
    key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not key: raise ValueError("Set OPENROUTER_API_KEY or pass --api-key")
    evaluate(args.model, key, args.split)
