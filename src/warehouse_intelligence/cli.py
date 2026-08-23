from __future__ import annotations

import argparse
import json

from .evaluation import write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Smart warehouse intelligence benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="run the UCI Online Retail benchmark")
    benchmark.add_argument("--data-dir", default=".cache/online-retail")
    benchmark.add_argument("--output", default="evals/results/v0.1_uci_online_retail.json")
    benchmark.add_argument("--download", action="store_true")
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--sku-limit", type=int, default=250)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "benchmark":
        report = write_report(
            args.output,
            args.data_dir,
            download=args.download,
            seed=args.seed,
            sku_limit=args.sku_limit,
        )
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
