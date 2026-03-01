"""
make_sample_small.py — Generate a small synthetic demand sample for git.

Output: data/sample/demand_sample.csv
Columns: date, store_id, item_id, demand

Size: 10 stores × 50 items × 60 days = 30 000 rows (< 3 MB).
"""

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_DIR = Path(__file__).parent.parent / "data" / "sample"
OUTPUT_FILE = SAMPLE_DIR / "demand_sample.csv"

N_STORES = 10
N_ITEMS = 50
N_DAYS = 60
START_DATE = "2014-01-01"
RANDOM_SEED = 42


def generate_sample(
    n_stores: int = N_STORES,
    n_items: int = N_ITEMS,
    n_days: int = N_DAYS,
    start_date: str = START_DATE,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    dates = pd.date_range(start=start_date, periods=n_days, freq="D")
    store_ids = [f"STORE_{i:02d}" for i in range(1, n_stores + 1)]
    item_ids = [f"ITEM_{i:03d}" for i in range(1, n_items + 1)]

    records = []
    for store in store_ids:
        for item in item_ids:
            # Base demand with slight seasonality
            base = rng.uniform(5.0, 50.0)
            trend = rng.uniform(-0.05, 0.05)
            for day_idx, date in enumerate(dates):
                seasonal = 2.0 * np.sin(2 * np.pi * day_idx / 7)  # weekly cycle
                noise = rng.normal(0, base * 0.1)
                demand = max(0.0, base + trend * day_idx + seasonal + noise)
                # Occasionally inject a demand shock
                if rng.random() < 0.02:
                    demand *= rng.uniform(3.0, 6.0)
                records.append(
                    {
                        "date": date.strftime("%Y-%m-%d"),
                        "store_id": store,
                        "item_id": item,
                        "demand": round(demand, 4),
                    }
                )

    df = pd.DataFrame(records)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a small demand sample CSV")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"Output CSV path. Default: {OUTPUT_FILE}",
    )
    parser.add_argument("--n-stores", type=int, default=N_STORES)
    parser.add_argument("--n-items", type=int, default=N_ITEMS)
    parser.add_argument("--n-days", type=int, default=N_DAYS)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample(
        n_stores=args.n_stores,
        n_items=args.n_items,
        n_days=args.n_days,
    )
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows to {args.output}")


if __name__ == "__main__":
    main()
