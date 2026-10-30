"""Write the markdown results table for a run: python -m eval.report eval/results/scifact.json"""

import json
import sys
from pathlib import Path
from typing import Any

DOCS = Path(__file__).parent.parent / "docs" / "results"
METRICS = ["recall@10", "recall@100", "mrr@10", "ndcg@10"]


def table(summary: dict[str, Any]) -> str:
    lines = [
        "| Method | " + " | ".join(METRICS) + " | p50 ms | p95 ms | Rescue | Damage |",
        "|---" * (len(METRICS) + 5) + "|",
    ]
    for method, s in summary.items():
        cells = [f"{s[m]:.3f}" for m in METRICS] + [str(s["p50_ms"]), str(s["p95_ms"])]
        cells += [f"{s[m]:.1%}" if m in s else "" for m in ["rescue", "damage"]]
        lines.append(f"| `{method}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(path: str) -> None:
    run = json.loads(Path(path).read_text())
    meta = run["meta"]
    header = (
        f"# {meta['dataset']}\n\n"
        f"{meta['queries']} queries, {meta['documents']} documents, run {meta['date']}. "
        f"Dense `{meta['dense_model']}`, sparse `{meta['sparse_model']}`, "
        f"rerank `{meta['cross_encoder']}` over the top {meta['k']}. "
        "Latency is in-process on CPU, one query at a time.\n\n"
    )
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / f"{meta['dataset']}.md"
    out.write_text(header + table(run["summary"]) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1])
