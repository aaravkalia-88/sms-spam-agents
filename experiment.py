"""Train and compare two PyTorch agents for SMS spam classification."""

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

AGENTS = ("word", "character")
WIDTH = 4096
STEPS = 45
MAX_MESSAGE_LENGTH = 4096
BOOTSTRAP_SAMPLES = 10_000
LABELS = {"ham": 0, "spam": 1}
CONTACT = re.compile(r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.\w+")
WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Message:
    row: int
    text: str
    label: int


def normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = CONTACT.sub(" contacttoken ", text)
    return " ".join(re.sub(r"\d+", "0", text).split())


def read_dataset(path):
    if path.stat().st_size > 2_000_000:
        raise ValueError("Expected a Kaggle SMS CSV under 2 MB.")

    groups = defaultdict(list)
    with path.open(encoding="latin-1", newline="") as file:
        rows = csv.DictReader(file, strict=True)
        if not {"v1", "v2"}.issubset(rows.fieldnames or []):
            raise ValueError("The Kaggle CSV must have v1 (label) and v2 (message) columns.")
        for row_number, row in enumerate(rows, start=1):
            text, label = row["v2"], row["v1"]
            if not text or len(text) > MAX_MESSAGE_LENGTH or label not in LABELS:
                raise ValueError(f"Row {row_number} has an invalid label or message.")
            key = " ".join(WORD.findall(normalized(text)))
            groups[key].append(Message(row_number, text, LABELS[label]))

    messages = [group[0] for group in groups.values() if len({m.label for m in group}) == 1]
    raw = sum(map(len, groups.values()))
    return messages, {
        "raw": raw,
        "discards": raw - len(messages),
        "conflicting_groups": len(groups) - len(messages),
    }


def split_dataset(messages, seed):
    """Within each label, place 15% (rounded down) in each held-out split."""
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for label in LABELS.values():
        subset = [message for message in messages if message.label == label]
        held_out = len(subset) * 15 // 100
        if held_out == 0:
            raise ValueError("The dataset needs at least seven unique messages in each class.")
        rng.shuffle(subset)
        test.extend(subset[:held_out])
        validation.extend(subset[held_out : 2 * held_out])
        train.extend(subset[2 * held_out :])
    return train, validation, test


def encode(texts, agent):
    """Map hashed n-grams to binary features with unit row norm."""
    features = torch.zeros((len(texts), WIDTH))
    for row, message in enumerate(texts):
        clean = normalized(message)
        if agent == "word":
            grams = set(WORD.findall(clean))
        else:
            grams = {clean[i : i + n] for n in (3, 4, 5) for i in range(len(clean) - n + 1)}
        columns = {
            int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=4).digest(), "big") % WIDTH
            for gram in grams
        }
        if columns:
            features[row, list(columns)] = 1 / math.sqrt(len(columns))
    return features


def fit(features, labels):
    model = nn.Linear(WIDTH, 1)
    positive = labels.sum()
    loss = nn.BCEWithLogitsLoss(pos_weight=((len(labels) - positive) / positive).sqrt())
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.01)
    for _ in range(STEPS):
        optimizer.zero_grad()
        loss(model(features).squeeze(1), labels).backward()
        optimizer.step()
    return model.eval()


@torch.inference_mode()
def score(model, features):
    return model(features).squeeze(1).sigmoid()


def coordinator(word, character):
    return (word + character) / 2


def evaluate(scores, labels, threshold):
    predicted = scores >= threshold
    positive = labels.bool()
    tp = int((predicted & positive).sum())
    fp = int((predicted & ~positive).sum())
    tn = int((~predicted & ~positive).sum())
    fn = int((~predicted & positive).sum())
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": (tp + tn) / len(labels),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
    }


def select_threshold(scores, labels):
    candidates = range(5, 100, 5)
    best = max(candidates, key=lambda t: (evaluate(scores, labels, t / 100)["f1"], -abs(t - 50)))
    return best / 100


def paired_bootstrap(labels, baseline, coordinator_predictions, *, seed):
    """Estimate the coordinator's F1 difference on paired held-out predictions."""
    pair_codes = 2 * baseline + coordinator_predictions
    counts = np.stack(
        [
            np.bincount(pair_codes[~labels], minlength=4),
            np.bincount(pair_codes[labels], minlength=4),
        ]
    )

    def spam_f1(ham, spam, positive_cells):
        true_positives = spam[..., positive_cells].sum(axis=-1)
        false_positives = ham[..., positive_cells].sum(axis=-1)
        return 2 * true_positives / (spam.sum(axis=-1) + true_positives + false_positives)

    # Resample paired outcomes within each actual label to keep class counts fixed.
    rng = np.random.default_rng(seed % (2**64))
    ham_samples, spam_samples = [
        rng.multinomial(int(row.sum()), row / row.sum(), size=BOOTSTRAP_SAMPLES) for row in counts
    ]
    ham, spam = counts
    delta = spam_f1(ham, spam, [1, 3]) - spam_f1(ham, spam, [2, 3])
    sampled_deltas = spam_f1(ham_samples, spam_samples, [1, 3]) - spam_f1(
        ham_samples, spam_samples, [2, 3]
    )
    buckets = ("both_ham", "coordinator_only_spam", "baseline_only_spam", "both_spam")
    return {
        "delta_f1": float(delta),
        "ci95": np.quantile(sampled_deltas, [0.025, 0.975]).tolist(),
        "resamples": BOOTSTRAP_SAMPLES,
        "seed": seed,
        "pairs": {
            label: dict(zip(buckets, row.tolist(), strict=True))
            for label, row in zip(LABELS, counts, strict=True)
        },
    }


def train(args):
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rows, counts = read_dataset(args.csv)
    training, validation, test = split_dataset(rows, args.seed)
    splits = {"train": training, "validation": validation, "test": test}
    labels = {
        name: torch.tensor([m.label for m in part], dtype=torch.float32)
        for name, part in splits.items()
    }
    states, validation_scores, test_scores = {}, {}, {}
    for agent in AGENTS:
        model = fit(encode([m.text for m in training], agent), labels["train"])
        states[agent] = model.state_dict()
        validation_scores[agent] = score(model, encode([m.text for m in validation], agent))
        test_scores[agent] = score(model, encode([m.text for m in test], agent))

    validation_scores["coordinator"] = coordinator(*(validation_scores[a] for a in AGENTS))
    test_scores["coordinator"] = coordinator(*(test_scores[a] for a in AGENTS))
    thresholds = {
        name: select_threshold(scores, labels["validation"])
        for name, scores in validation_scores.items()
    }
    majority_label = int(labels["train"].mean() > 0.5)
    metrics = {
        "majority": evaluate(torch.full_like(labels["test"], majority_label), labels["test"], 0.5)
    }
    for name, scores in test_scores.items():
        metrics[name] = {
            "threshold": thresholds[name],
            **evaluate(scores, labels["test"], thresholds[name]),
        }

    predicted = {name: (scores >= thresholds[name]).numpy() for name, scores in test_scores.items()}
    actual = labels["test"].bool().numpy()

    missed = [
        i
        for i, m in enumerate(test)
        if m.label and test_scores["coordinator"][i] < thresholds["coordinator"]
    ]
    failure = None
    if missed:
        # Prefer a missed spam message where the two agents disagree most.
        i = max(
            missed, key=lambda i: abs(float(test_scores["word"][i] - test_scores["character"][i]))
        )
        failure = {
            "data_row": test[i].row,
            "expected": "spam",
            "prediction": "ham",
            "scores": {name: float(scores[i]) for name, scores in test_scores.items()},
        }

    report = {
        "dataset": {**counts, "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest()},
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "os": platform.system(),
            "architecture": platform.machine(),
            "device": "cpu",
        },
        "seed": args.seed,
        "features_per_agent": WIDTH,
        "training_steps": STEPS,
        "splits": {
            name: {"total": len(part), "spam": int(labels[name].sum())}
            for name, part in splits.items()
        },
        "models": metrics,
        "paired_f1": {
            agent: paired_bootstrap(
                actual, predicted[agent], predicted["coordinator"], seed=args.seed
            )
            for agent in AGENTS
        },
        "failure": failure,
    }
    torch.save({"states": states, "threshold": thresholds["coordinator"]}, args.model)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved model to {args.model} and results to {args.report}.")


def predict(args):
    message = sys.stdin.read(MAX_MESSAGE_LENGTH + 1).strip()
    if not 1 <= len(message) <= MAX_MESSAGE_LENGTH:
        raise ValueError(
            f"Pass one nonempty message of at most {MAX_MESSAGE_LENGTH} characters on stdin."
        )

    checkpoint = torch.load(args.model, weights_only=True, map_location="cpu")
    scores = {}
    for agent in AGENTS:
        model = nn.Linear(WIDTH, 1)
        model.load_state_dict(checkpoint["states"][agent])
        scores[agent] = score(model.eval(), encode([message], agent))
    combined = float(coordinator(*(scores[agent][0] for agent in AGENTS)))
    threshold = checkpoint["threshold"]
    print(
        json.dumps(
            {
                "label": "spam" if combined >= threshold else "ham",
                "score": combined,
                "threshold": threshold,
                "agents": {agent: scores[agent].item() for agent in AGENTS},
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser(
        "train", help="Train on Kaggle CSV and evaluate on a held-out split."
    )
    training.add_argument("--csv", type=Path, required=True)
    training.add_argument("--model", type=Path, default=Path("model.pt"))
    training.add_argument("--report", type=Path, default=Path("results.json"))
    training.add_argument("--seed", type=int, default=42)
    prediction = commands.add_parser("predict", help="Classify one message from stdin.")
    prediction.add_argument("--model", type=Path, default=Path("model.pt"))
    args = parser.parse_args()
    try:
        if args.command == "train":
            if len({args.csv.resolve(), args.model.resolve(), args.report.resolve()}) != 3:
                raise ValueError("Use distinct paths for the input CSV, model, and results.")
            train(args)
        else:
            predict(args)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
