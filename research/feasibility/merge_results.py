"""Merge multiple shard result JSONs into one and print a summary."""
import json
import sys
import os
from dataclasses import dataclass
from typing import Optional

def main():
    if len(sys.argv) < 2:
        print("Usage: python merge_results.py result1.json result2.json ...")
        sys.exit(1)

    all_results = []
    model = None
    max_context = None
    for path in sys.argv[1:]:
        data = json.load(open(path))
        model = model or data.get("model")
        max_context = max_context or data.get("max_context_tokens")
        all_results.extend(data["results"])

    passed = [r for r in all_results if r["success"]]
    failed = [r for r in all_results if not r["success"]]
    tps_vals = [r["tps"] for r in all_results if r["tps"] > 0]
    avg_tps = sum(tps_vals) / len(tps_vals) if tps_vals else 0
    avg_tokens = sum(r["tokens_used"] for r in all_results) / len(all_results)

    print(f"Model:  {model}")
    print("=" * 60)
    print(f"RESULTS: {len(passed)}/{len(all_results)} passed")
    print(f"Avg TPS:    {avg_tps:.1f}")
    print(f"Avg tokens: {avg_tokens:.0f} / {max_context}")

    if failed:
        print(f"\nFailed:")
        for r in failed:
            print(f"  [{r['category']}] {r['task']}: {r.get('error') or 'wrong answer'}")

    categories = sorted(set(r["category"] for r in all_results))
    print("\nBy category:")
    for cat in categories:
        cat_results = [r for r in all_results if r["category"] == cat]
        cat_passed = sum(1 for r in cat_results if r["success"])
        print(f"  {cat:<20} {cat_passed}/{len(cat_results)}")
    print("=" * 60)

    # Save merged file
    out = os.path.join(os.path.dirname(sys.argv[1]), f"merged_{int(__import__('time').time())}.json")
    json.dump({"model": model, "max_context_tokens": max_context, "results": all_results}, open(out, "w"), indent=2)
    print(f"\nMerged results saved to {out}")

if __name__ == "__main__":
    main()
