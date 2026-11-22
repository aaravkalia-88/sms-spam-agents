# SMS Spam Agents

[![Tests](https://github.com/aaravkalia-88/sms-spam-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/aaravkalia-88/sms-spam-agents/actions/workflows/ci.yml)

A PyTorch benchmark by **Aarav Kalia**. Send one SMS to two independent classifiers: a word agent and a character agent. A coordinator averages their scores and returns `spam` or `ham` (legitimate).

**Research question:** Does combining word and character cues improve spam detection over either classifier alone? The coordinator won on one of three seeded splits. On the primary split, it caught two more spam messages than the word agent and flagged four more legitimate messages by mistake.

## Held-out results

Kaggle SMS Spam Collection, seed 42. The held-out set contained **763 messages**, including 89 spam messages. Precision, recall and F1 refer to the spam class.

| System | Accuracy | Precision | Recall | F1 | False positives | Missed spam |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Always predict ham | 88.34% | 0.00% | 0.00% | 0.00% | 0 | 89 |
| Word agent | **98.03%** | **95.12%** | 87.64% | **91.23%** | 4 | 11 |
| Character agent | 97.12% | 86.02% | **89.89%** | 87.91% | 13 | 9 |
| Two-agent coordinator | 97.77% | 90.91% | **89.89%** | 90.40% | 8 | 9 |

| Seed | Word F1 | Character F1 | Coordinator F1 |
| --- | ---: | ---: | ---: |
| [42](results/seed-42.json) | **91.23%** | 87.91% | 90.40% |
| [17](results/seed-17.json) | 88.27% | 89.02% | **90.80%** |
| [73](results/seed-73.json) | 89.77% | **93.71%** | 92.22% |

### One observed failure

The coordinator missed a cash-giveaway promotion in seed 42 (Kaggle data row 4753, excluding the header). The word agent scored it **0.214**, the character agent **0.479**. The character score cleared its own threshold of 0.40. Averaging brought the coordinator score to **0.347**, just below its 0.35 cutoff. The miss shows how equal weighting can suppress a specialist's correct prediction.

## Method

1. Read `v1` (label) and `v2` (message) from the Kaggle CSV. Of 5,572 data rows, 5,097 remained after grouping by normalized text and removing repeats before splitting. Normalization folds case, replaces web and email contacts and digit sequences, and ignores punctuation and whitespace in the grouping key.
2. Stratify by spam and ham. Use about 70% of each class for training, 15% for validation and 15% for testing. Each seeded split contains 3,571 training, 763 validation and 763 test messages.
3. Give the word agent unigram features and the character agent 3–5 character n-grams. Each trains its own PyTorch linear classifier on 4,096 hashed features. Both use class-weighted binary cross-entropy and the same optimizer settings.
4. Use an equal-weight average for the coordinator. Select each model's cutoff using validation F1 on a fixed grid, then evaluate those cutoffs on the test set. The JSON files include the full confusion matrices, seeds, environment and dataset checksum.

These are three overlapping hold-outs from one historical English-language corpus. Similar messages can still land on different sides of a split when their normalized text differs. The coordinator uses two classifiers' parameters, and the scores are uncalibrated. The experiment measures within-corpus performance. An external, newer dataset would be needed to estimate performance on current messages.

## Reproduce and try it

The reported runs used Python 3.14, PyTorch 2.15 and a macOS arm64 CPU. No GPU, API key or service is needed for training or prediction. On Linux, install the [PyTorch CPU wheel](https://pytorch.org/get-started/locally/) first with `python -m pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu`.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt

mkdir -p data artifacts
curl -fL 'https://www.kaggle.com/api/v1/datasets/download/uciml/sms-spam-collection-dataset?datasetVersionNumber=1' -o data/sms-spam.zip
unzip -p data/sms-spam.zip spam.csv > data/spam.csv

for seed in 42 17 73; do
  python experiment.py train --csv data/spam.csv --seed "$seed" \
    --model "artifacts/model-$seed.pt" --report "artifacts/seed-$seed.json"
done

printf '%s\n' 'Congratulations! Claim your free cash prize now.' |
  python experiment.py predict --model artifacts/model-42.pt
```

The prediction includes the final label, each specialist's score and the coordinator cutoff. Expected label for the synthetic demo above: `spam`.

## Software tests

**7 tests passed with 97.44% combined statement and branch coverage** on the local macOS run. Tests check CSV validation, duplicate handling, stratification, feature encoding, scoring and the train/predict command line. They use synthetic fixtures, so they run without downloading the dataset.

```sh
ruff check .
ruff format --check .
python -m pytest --cov=experiment --cov-fail-under=80
```

## Data and license

The benchmark reads version 1 of the [Kaggle SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset), a mirror of the corpus by Tiago Almeida and José María Gómez Hidalgo. The [original UCI dataset](https://doi.org/10.24432/C5CC84) is credited under CC BY 4.0. This repo contains source code and aggregate results. The [MIT license](LICENSE) covers the project code.
