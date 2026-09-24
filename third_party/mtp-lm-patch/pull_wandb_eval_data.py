"""Pull wandb eval-run summaries into a CSV (hop 4).

No metric-specific handling is needed for controlled-rollout metrics: the
summary is taken wholesale, so every "{task}/controlled/..." key the pusher
logs arrives automatically. If controlled metrics are missing from the CSV the
bug is at hop 3 (push_lmeval_metrics_to_wandb.py), not here.
"""

import argparse
import math
import numbers
import os
import time

import pandas as pd
import wandb
from tqdm import tqdm


def _is_non_finite(value):
    """True for NaN/inf numerics, False for anything non-numeric (incl. bools)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Number):
        return False
    try:
        return not math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entity", default="po-wei-huang-university-of-oxford")
    parser.add_argument("--project", default="singleshot-evals")
    parser.add_argument(
        "--out_dir",
        default="figures_data_raw",
        help="created if missing; a wrong CWD used to raise or write somewhere unread",
    )
    parser.add_argument(
        "--write_latest",
        action="store_true",
        default=True,
        help="also write <project>_summary_table_latest.csv beside the timestamped file",
    )
    parser.add_argument("--no_write_latest", dest="write_latest", action="store_false")
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    summary_list, config_list, name_list, tag_list = [], [], [], []
    for run in tqdm(runs, desc="Pulling wandb eval data"):
        # .summary holds the metric values; ._json_dict omits large files.
        summary = run.summary._json_dict

        # A non-finite value silently drops the whole row at hop 5, so name it
        # here -- this catches a hop-3 regression immediately rather than as
        # missing points on a plot.
        offenders = [k for k, v in summary.items() if _is_non_finite(v)]
        if offenders:
            print(
                f"WARNING: run {run.name!r} has non-finite summary value(s): "
                + ", ".join(sorted(offenders))
            )

        summary_list.append(summary)
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )
        name_list.append(run.name)
        tag_list.append(run.tags)

    runs_df = pd.DataFrame(
        {
            "summary": summary_list,
            "config": config_list,
            "name": name_list,
            "tags": tag_list,
        }
    )

    os.makedirs(args.out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    stamped = os.path.join(
        args.out_dir, f"{args.project}_summary_table_{timestamp}.csv"
    )
    runs_df.to_csv(stamped)
    print(f"wrote {stamped}  ({len(runs_df)} runs)")

    if args.write_latest:
        latest = os.path.join(args.out_dir, f"{args.project}_summary_table_latest.csv")
        runs_df.to_csv(latest)
        print(f"wrote {latest}")


if __name__ == "__main__":
    main()
