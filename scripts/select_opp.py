#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mars_surrogate.models import SurrogateModel
from mars_surrogate.feasibility import StrictFeasibilityClassifier
from mars_surrogate.schema import OPP_COLUMNS
from mars_surrogate.selector import ModeAwareSelector


def select_from_args(args: argparse.Namespace) -> dict[str, Any]:
    model = SurrogateModel.load(args.model_dir)
    feasibility_model = None
    if args.selector_mode == "risk":
        feasibility_model = StrictFeasibilityClassifier.load(args.model_dir)
    candidates = pd.read_csv(args.candidate_csv)[OPP_COLUMNS]
    selector = ModeAwareSelector(
        model,
        candidates,
        feasibility_model=feasibility_model,
        safety_margin_ms=args.safety_margin_ms,
    )
    metadata: dict[str, Any] = {
        "text_length_target": args.text_length_target,
        "num_views": args.num_views,
        "denoising_steps": args.denoising_steps,
    }
    if args.actual_text_words is not None:
        metadata["actual_text_words"] = args.actual_text_words
    if args.input_token_count is not None:
        metadata["input_token_count"] = args.input_token_count
    result = selector.select_tail_aware(
        metadata,
        deadline_ms=args.deadline_ms,
        mode=args.selector_mode,
        feasible_prob_threshold=args.feasible_prob_threshold,
    )
    result.pop("candidate_predictions", None)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Select an Eco OPP using trained surrogate predictions.")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--text-length-target", type=float, required=True)
    parser.add_argument("--actual-text-words", type=float, default=None)
    parser.add_argument("--input-token-count", type=float, default=None)
    parser.add_argument("--num-views", type=float, default=2)
    parser.add_argument("--denoising-steps", type=float, required=True)
    parser.add_argument("--deadline-ms", type=float, required=True)
    parser.add_argument("--safety-margin-ms", type=float, default=0.0)
    parser.add_argument("--selector-mode", choices=["median_margin", "tail", "risk"], default="median_margin")
    parser.add_argument("--feasible-prob-threshold", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(select_from_args(args), indent=2))


if __name__ == "__main__":
    main()
