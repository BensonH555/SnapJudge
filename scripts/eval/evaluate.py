"""
Evaluate detector predictions on the test set at a fixed false-positive rate.

Reads prediction files written by scripts/inference.py or scripts/inference_image.py. A passage is classified
as AI-generated if score >= threshold. For proofs and prose separately, prints the detection rate of every
generator family and the Frontier and Open averages (unweighted means over the families).

Usage:
  python scripts/eval/evaluate.py --predictions latex_predictions.jsonl --detector latex --fpr 0.1%
"""

import argparse
import json

import numpy as np
import pandas as pd

# Thresholds of the released detectors (see README, "Thresholds")
THRESHOLDS = {
    "latex": {"0.1%": 0.9986114501953125, "0.05%": 0.9994887113571167, "0.01%": 0.9999279816031456, "0.005%": 0.9999589702010154, "0.001%": 0.9999960217535496},
    "text": {"0.1%": 0.9995121955871582, "0.05%": 0.9997367536425591, "0.01%": 0.9999424217402937, "0.005%": 0.9999660922855138, "0.001%": 0.9999893260324001},
    "image": {"0.1%": 0.9788680946826935, "0.05%": 0.9905976241827011, "0.01%": 0.9981755863428113, "0.005%": 0.9991849524974823, "0.001%": 0.9998685571908948},
}
FRONTIER = ["GPT-4.1", "GPT-5.1", "o4-mini", "Claude Sonnet 5", "Gemini 3.1 Pro"]


def load_predictions(path):
    """.jsonl or .csv prediction file; all values are kept as strings."""
    if path.endswith(".csv"):
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    with open(path) as f:
        return pd.DataFrame([json.loads(line, parse_float=str, parse_int=str) for line in f])


def evaluate(df, threshold):
    """Detection rate per family and Frontier / Open averages."""
    scores = np.array([float(s) for s in df["score"]])
    is_ai = (df["label_id"] == "1").values
    flagged = scores >= threshold
    families = df["family"].values
    open_weight = sorted(set(families[is_ai]) - set(FRONTIER))
    cells = {fam: flagged[is_ai & (families == fam)].mean() for fam in FRONTIER + open_weight}
    cells["Frontier avg."] = np.mean([cells[fam] for fam in FRONTIER])
    cells["Open avg."] = np.mean([cells[fam] for fam in open_weight])
    return cells


def main():
    parser = argparse.ArgumentParser(description="Evaluate detector predictions at a fixed false-positive rate")
    parser.add_argument("--predictions", required=True, help="Prediction file (.jsonl or .csv)")
    parser.add_argument("--detector", choices=THRESHOLDS, help="Use the threshold of this released detector")
    parser.add_argument("--fpr", default="0.1%", choices=["0.1%", "0.05%", "0.01%", "0.005%", "0.001%"])
    parser.add_argument("--threshold", type=float, help="Threshold instead of --detector")
    args = parser.parse_args()
    if args.threshold is None and not args.detector:
        parser.error("give --detector or --threshold")

    threshold = args.threshold if args.threshold is not None else THRESHOLDS[args.detector][args.fpr]
    df = load_predictions(args.predictions)
    print("Threshold:", threshold)

    for register in ("proof", "prose"):
        sub = df[df["register"] == register]
        cells = evaluate(sub, threshold)
        print(f"\n{register}: {(sub['label_id'] == '1').sum()} AI-generated, {(sub['label_id'] == '0').sum()} human-written")
        for key, value in cells.items():
            print(f"  {key:<32} {value:.4f}")

if __name__ == "__main__":
    main()
