import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from experiment import (
    Message,
    coordinator,
    encode,
    evaluate,
    paired_bootstrap,
    read_dataset,
    select_threshold,
    split_dataset,
)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="latin-1") as file:
        writer = csv.writer(file)
        writer.writerow(["v1", "v2"])
        writer.writerows(rows)


def test_split_drops_duplicates_and_conflicts_before_splitting(tmp_path):
    path = tmp_path / "spam.csv"
    rows = [("ham", f"Dinner plan {letter}") for letter in "abcdefghij"]
    rows += [("spam", f"Claim prize {letter}") for letter in "abcdefghij"]
    rows += [("spam", "  CLAIM prize A! "), ("ham", "Claim prize b.")]
    write_csv(path, rows)

    messages, counts = read_dataset(path)
    splits = split_dataset(messages, seed=17)
    assert counts == {"raw": 22, "discards": 3, "conflicting_groups": 1}
    assert splits == split_dataset(messages, seed=17)
    assert [{message.label for message in part} for part in splits] == [{0, 1}] * 3
    identities = [message.row for part in splits for message in part]
    assert len(identities) == len(set(identities)) == 19
    assert 12 not in identities


def test_invalid_label_and_small_split_are_rejected(tmp_path):
    path = tmp_path / "spam.csv"
    write_csv(path, [("unknown", "private input")])
    with pytest.raises(ValueError, match=r"Row 1") as error:
        read_dataset(path)
    assert "private input" not in str(error.value)
    with pytest.raises(ValueError, match="each class"):
        split_dataset([Message(1, "hello", 0)], seed=3)


def test_dataset_requires_the_kaggle_columns(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("target,text\nspam,Example\n")
    with pytest.raises(ValueError, match=r"v1 .* v2"):
        read_dataset(path)


@pytest.mark.parametrize("agent", ["word", "character"])
def test_encoding_is_deterministic_and_normalizes_contact_details(agent):
    scores = encode(["Call 07123456789", "CALL 07987654321", ""], agent)
    assert torch.equal(scores[0], scores[1])
    assert torch.linalg.vector_norm(scores[0]).item() == pytest.approx(1.0)
    assert torch.count_nonzero(scores[2]).item() == 0


def test_coordinator_metrics_and_validation_threshold():
    probabilities = coordinator(
        torch.tensor([0.1, 0.8, 0.7, 0.4]), torch.tensor([0.1, 0.8, 0.7, 0.4])
    )
    labels = torch.tensor([0, 1, 0, 1])
    metrics = evaluate(probabilities, labels, threshold=0.5)
    assert {key: metrics[key] for key in ("tp", "tn", "fp", "fn")} == {
        "tp": 1,
        "tn": 1,
        "fp": 1,
        "fn": 1,
    }
    assert metrics["f1"] == 0.5
    assert select_threshold(probabilities, labels) == 0.4
    torch.testing.assert_close(
        coordinator(torch.tensor([0.3]), torch.tensor([0.7])), torch.tensor([0.5])
    )


@pytest.mark.parametrize("seed", [42, -1])
def test_paired_bootstrap_agreement_has_zero_width_interval(seed):
    labels = np.array([False, False, True, True])
    predicted = np.array([False, True, True, False])
    result = paired_bootstrap(labels, predicted, predicted, seed=seed)
    assert result["delta_f1"] == 0
    assert result["ci95"] == [0, 0]


def test_paired_bootstrap_counts_each_kind_of_disagreement():
    labels = np.array([False] * 4 + [True] * 4)
    baseline = np.array([False, False, True, False, False, True, True, True])
    candidate = np.array([False, True, False, False, True, True, True, True])
    result = paired_bootstrap(labels, baseline, candidate, seed=42)
    assert result == paired_bootstrap(labels, baseline, candidate, seed=42)
    assert result["delta_f1"] == pytest.approx(8 / 9 - 6 / 8)
    assert result["pairs"]["ham"] == {
        "both_ham": 2,
        "coordinator_only_spam": 1,
        "baseline_only_spam": 1,
        "both_spam": 0,
    }
    assert result["pairs"]["spam"]["coordinator_only_spam"] == 1
    assert result["ci95"][0] < result["delta_f1"] < result["ci95"][1]


def test_train_and_predict_entry_points(tmp_path):
    path = tmp_path / "spam.csv"
    write_csv(
        path,
        [
            (label, f"{prefix} {letter}")
            for label, prefix in [("ham", "Dinner with friends"), ("spam", "Claim cash prize now")]
            for letter in "abcdefghijklmnopqrst"
        ],
    )
    script = Path(__file__).with_name("experiment.py")
    model = tmp_path / "model.pt"
    report = tmp_path / "results.json"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "train",
            "--csv",
            str(path),
            "--model",
            str(model),
            "--report",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    results = json.loads(report.read_text())
    assert results["splits"]["test"]["total"] == 6
    assert set(results["models"]) == {"majority", "word", "character", "coordinator"}
    assert results["models"]["coordinator"]["f1"] == 1
    assert set(results["paired_f1"]) == {"word", "character"}
    assert results["paired_f1"]["word"]["resamples"] == 10_000
    assert sum(results["paired_f1"]["word"]["pairs"]["spam"].values()) == 3
    assert "Claim cash" not in report.read_text()

    prediction = subprocess.run(
        [
            sys.executable,
            str(script),
            "predict",
            "--model",
            str(model),
        ],
        input="Claim cash prize now!",
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(prediction.stdout)["label"] == "spam"

    blank = subprocess.run(
        [
            sys.executable,
            str(script),
            "predict",
            "--model",
            str(model),
        ],
        input=" ",
        check=False,
        capture_output=True,
        text=True,
    )
    assert blank.returncode != 0
    assert "message" in blank.stderr

    aliased_output = subprocess.run(
        [
            sys.executable,
            str(script),
            "train",
            "--csv",
            str(path),
            "--model",
            str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert aliased_output.returncode != 0
    assert "distinct paths" in aliased_output.stderr
    assert path.read_text().startswith("v1,v2")
