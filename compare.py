"""Score a results file against a reference results file with the brief's three checks.

    python compare.py reference.jsonl candidate.jsonl

Exits 1 when the candidate fails any threshold.
"""
import argparse
import json
import sys
from typing import TypedDict

F1_THRESHOLD = 0.95
EXACT_THRESHOLD = 0.85
WITHIN_ONE_THRESHOLD = 0.95

SecondKey = tuple[str, int]


class Second(TypedDict):
    objects: frozenset[str]
    people: int


def load(path: str) -> tuple[dict[SecondKey, Second], set[str]]:
    seconds: dict[SecondKey, Second] = {}
    errors: set[str] = set()
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "error" in row:
                errors.add(row["video"])
            else:
                seconds[(row["video"], row["second"])] = {"objects": frozenset(row["objects"]), "people": row["people"]}
    return seconds, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()

    ref, ref_errors = load(args.reference)
    cand, cand_errors = load(args.candidate)
    if not ref:
        print("reference has no seconds")
        return 1

    tp = fp = fn = exact = within_one = 0
    for key, r in ref.items():
        c = cand.get(key)
        objects = c["objects"] if c else frozenset[str]()
        tp += len(r["objects"] & objects)
        fn += len(r["objects"] - objects)
        fp += len(objects - r["objects"])
        if c is not None:
            exact += c["people"] == r["people"]
            within_one += abs(c["people"] - r["people"]) <= 1
    extra = cand.keys() - ref.keys()
    fp += sum(len(cand[k]["objects"]) for k in extra)

    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    exact_rate = exact / len(ref)
    within_rate = within_one / len(ref)
    passed = f1 >= F1_THRESHOLD and exact_rate >= EXACT_THRESHOLD and within_rate >= WITHIN_ONE_THRESHOLD

    print(f"seconds: reference {len(ref)}, candidate {len(cand)}, missing {len(ref.keys() - cand.keys())}, extra {len(extra)}")
    print(f"errors: reference {sorted(ref_errors)}, candidate {sorted(cand_errors)}")
    print(f"objects F1:        {f1:.4f}  (tp {tp}, fp {fp}, fn {fn}, need >= {F1_THRESHOLD})")
    print(f"people exact:      {exact_rate:.4f}  (need >= {EXACT_THRESHOLD})")
    print(f"people within 1:   {within_rate:.4f}  (need >= {WITHIN_ONE_THRESHOLD})")
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
