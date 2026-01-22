"""Run batch predictions for players aged 29-32."""
from __future__ import annotations

import argparse

from adapt_or_fade import run_batch_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predict player longevity")
    parser.add_argument("--data-url", default=None, help="GitHub raw CSV URL override")
    parser.add_argument("--output", default="batch_predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    if args.data_url:
        run_batch_predictions(data_url=args.data_url, output_path=args.output)
    else:
        run_batch_predictions(output_path=args.output)


if __name__ == "__main__":
    main()
