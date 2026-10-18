# EviKT Toolkit

**Evidential Learning and Uncertainty Evaluation for Knowledge Tracing**

EviKT Toolkit extends pyKT with Evidential Deep Learning (EDL) support for multiple Knowledge Tracing (KT) models. Given a student's response history, these models predict the probability of a correct response and estimate predictive uncertainty through an evidence-based distribution.

The toolkit provides data preprocessing, model training, independent testing, and uncertainty evaluation. It supports comparisons of predictive performance and analysis of whether models assign higher uncertainty to incorrect predictions.


## Features

- Eight supported models: DKT, AKT, AT-DKT, DKVMN, UKT, KeenKT, simpleKT, and RobustKT.
- Shared utilities for Softplus evidence, Dirichlet probabilities, and EDL losses.
- EDL and non-EDL training modes controlled by `--use_edl`.
- Data preprocessing and cross-validation fold construction inherited from pyKT.
- Training on training and validation data, followed by independent evaluation of saved checkpoints.
- AUC, ACC, MUD, and W2 reporting in EDL mode, with separate results for standard and windowed testing.

## Supported Models

The defaults below come from each model's training script. Set `--use_edl 1` or `--use_edl 0` explicitly to make the experiment configuration clear.

| Model | Training script (under `examples/`) | Default `emb_type` | EDL enabled by default |
| --- | --- | --- | --- |
| DKT | `wandb_dkt_train.py` | `qid` | No |
| AKT | `wandb_akt_train.py` | `qid` | No |
| AT-DKT | `wandb_atdkt_train.py` | `qid` | Yes |
| DKVMN | `wandb_dkvmn_train.py` | `qid` | No |
| UKT | `wandb_ukt_train.py` | `stoc_qid` | No |
| KeenKT | `wandb_keenkt_train.py` | `stoc_qid` | Yes |
| simpleKT | `wandb_simplekt_train.py` | `qid` | Yes |
| RobustKT | `wandb_robustkt_train.py` | `qid` | Yes |

KeenKT also exposes contrastive learning and diffusion modules through `--use_CL` and `--use_diffusion`, respectively. These options are model-specific; use the corresponding script's `--help` to inspect available arguments.

## Installation

In an isolated Python environment, run the following commands from the repository root:

```sh
python -m pip install -e .
python -c "import pykt; print(pykt.__file__)"
```

Confirm that the printed path points to this repository's `pykt` directory. Core dependencies declared in `setup.py` include PyTorch, NumPy, pandas, scikit-learn, and Weights & Biases (W&B). Training uses a GPU when CUDA is available and falls back to the CPU otherwise.

The repository does not currently provide a fully validated Python/PyTorch version combination or a dependency lockfile. Historical minimum versions in `setup.py` should not be interpreted as verified compatibility for the current code. Record your Python, PyTorch, CUDA, and dependency versions when reproducing experiments.

Run all preprocessing, training, and testing commands below from `examples`, because the scripts access `../configs/` through relative paths:

```sh
cd examples
```

## Data Preparation

### Using Preprocessed Data

Check the dataset entry in `configs/data_config.json`, especially:

- `dpath`: the data directory, relative to the working directory used to run the script.
- `num_c`, `num_q`, and `input_type`: the numbers of concepts and questions, and the input fields.
- `train_valid_file`, `test_file`, and `test_window_file`: the training/validation, standard test, and windowed test files.
- `folds` and `maxlen`: fold identifiers and sequence length settings.

For `assist2015`, the default directory is `../data/assist2015`, with these main sequence files:

```text
data/assist2015/
├── train_valid_sequences.csv
├── test_sequences.csv
└── test_window_sequences.csv
```

If these files are already available and match the configuration, proceed directly to training. A dataset entry in the configuration does not mean that its data files have been downloaded.

### Preprocessing Raw Data

Obtain the raw files for the dataset you intend to use. For ASSISTments 2015, place the raw file at:

```text
data/assist2015/2015_100_skill_builders_main_problems.csv
```

Then run the following command from `examples`:

```sh
python data_preprocess.py --dataset_name assist2015 --min_seq_len 3 --maxlen 200 --kfold 5
```

Preprocessing generates sequence files and updates `configs/data_config.json`. Raw file paths for other datasets are defined in `dname2paths` in `examples/data_preprocess.py`. The `--file_path` argument currently applies only to the `peiyou` branch.

The preprocessing script invokes the shell command `rm` to remove old `.pkl` caches. This command may be unavailable on native Windows. When reprocessing an existing dataset in that environment, separately remove stale `.pkl` caches from that dataset's directory to avoid loading outdated data.

## Training and Testing

### 1. Train DKT with EDL

Run from `examples`:

```sh
python wandb_dkt_train.py --dataset_name assist2015 --use_edl 1 --use_wandb 0 --fold 0 --seed 42 --save_dir saved_model
```

`--fold 0` selects fold 0 for validation and uses the remaining folds for training. This command runs one validation fold, not all cross-validation folds. The test set is not used during training.

Training creates a subdirectory under `saved_model`, named using the experiment parameters. The logs print the actual `save_dir` and checkpoint path. The experiment directory contains:

```text
saved_model/<experiment_directory>/
├── config.json
└── qid_model.ckpt
```

`config.json` stores training, model, and data configurations, along with run arguments. Checkpoints are named `<emb_type>_model.ckpt`; UKT and KeenKT use `stoc_qid_model.ckpt` by default.

### 2. Evaluate a Saved Model

Replace the path below with the actual experiment subdirectory printed during training, keeping the quotation marks around it:

```sh
python wandb_predict.py --save_dir "saved_model/REPLACE_WITH_EXPERIMENT_DIRECTORY" --use_wandb 0 --skip_question_eval 1
```

Evaluation restores the model and EDL settings from the saved configuration, so no additional `--use_edl` argument is needed. Dataset paths are still read from the current `configs/data_config.json`; ensure that this configuration remains consistent with the training data.

EDL mode evaluates concept-level predictions and automatically skips question-level fusion. The example's `--skip_question_eval 1` also prevents question-level test datasets from being loaded.

### 3. Compare Other Configurations

Train DKT without EDL:

```sh
python wandb_dkt_train.py --dataset_name assist2015 --use_edl 0 --use_wandb 0 --fold 0 --seed 42 --save_dir saved_model_bce
```

Train UKT with EDL:

```sh
python wandb_ukt_train.py --dataset_name assist2015 --use_edl 1 --use_wandb 0 --fold 0 --seed 42 --save_dir saved_model_ukt_edl
```

When comparing models or training modes, keep dataset splits, validation folds, and random seeds consistent. Record model hyperparameters and any additional module settings.

### Common Arguments and Configuration

| Argument / configuration | Description |
| --- | --- |
| `--dataset_name` | Dataset name in `configs/data_config.json` |
| `--use_edl` | `1` enables EDL; `0` selects the non-EDL branch |
| `--edl_annealing_step` | Number of epochs over which the EDL KL regularization weight increases to 1 |
| `--fold` | Validation fold identifier |
| `--seed` | Random seed |
| `--save_dir` | Output root for training; specific experiment directory for testing |
| `--use_wandb` | `0` disables W&B logging; `1` enables it |
| `train_config` in `configs/kt_config.json` | Base epoch count, batch size, and optimizer settings |

Model hyperparameters primarily come from training script defaults and command-line arguments. Editing a model-specific block in `kt_config.json` does not necessarily override those arguments. Some models override the base batch size, and some scripts expose `--batch_size` and `--num_epochs`; refer to the corresponding script for its behavior.

All examples disable W&B. To enable it, configure authentication. The prediction script also reads `api_key` from `configs/wandb.json`. Do not commit real API keys to the repository.

## EDL and Evaluation Metrics

The current implementation uses binary response classification: class `0` is an incorrect student response, and class `1` is a correct student response. Shared computations are implemented in `pykt/models/edl.py`:

```text
evidence = softplus(logits)
alpha = evidence + 1
S = sum(alpha)
p = alpha / S
u = 2 / S
```

The sum is taken over the class dimension. For an individual prediction, `p[1]` is the probability of a correct student response, and `u` is evidential uncertainty. Lower total evidence produces higher uncertainty. The EDL loss utilities provide expected MSE/BCE prediction terms and an annealed KL regularization term; whether loss selection is exposed as a command-line option depends on the model script.

In the metrics below, a *correct or incorrect prediction* refers to whether the model's predicted label matches the true label. This is distinct from whether the student answered the question correctly.

| Metric | Definition | Interpretation |
| --- | --- | --- |
| AUC | Area under the ROC curve | Measures discrimination between correct and incorrect student responses |
| ACC | Prediction accuracy, using `p[1] >= 0.5` to predict a correct response | Measures agreement between predicted and true labels |
| MUD | Mean uncertainty of incorrect predictions minus that of correct predictions | A positive value means incorrect predictions are more uncertain on average |
| W2 | Second-order Wasserstein distance between the empirical uncertainty distributions of correct and incorrect predictions | Measures distributional separation without indicating its direction |

W2 is neither W1 nor the squared W2 distance. A larger W2 alone does not establish better uncertainty estimates; interpret it alongside MUD and predictive performance.

During EDL evaluation, MUD and W2 are reported as `NaN` if all predictions are correct or all are incorrect. AUC is reported as `NaN` if the true labels contain only one class.

Standard testing uses `test_file`, while windowed testing uses `test_window_file`. Metrics are computed separately because these files use different test sequence construction procedures.

## Output Files

After EDL evaluation, the main files in an experiment directory are:

```text
<experiment_directory>/
├── config.json
├── <emb_type>_model.ckpt
├── <emb_type>_final_test_metrics.csv
└── edl_analysis/
    ├── test_metrics.csv
    └── test_window_metrics.csv
```

- Both tables under `edl_analysis` contain the columns `auc,acc,mud,w2`.
- The final summary table contains standard and windowed test metrics, along with saved run arguments.
- The current EDL evaluation entry point does not export per-sample EDL CSVs, plots, or selective prediction results.

Non-EDL mode produces a final metrics summary and prediction TXT files, but does not generate the EDL analysis tables above. It also performs question-level fusion evaluation when question-level test files are configured and question-level evaluation is enabled.

## Repository Structure

```text
.
├── configs/                 # Dataset, training, and experiment configurations
├── data/                    # Local datasets
├── examples/                # Preprocessing, training, testing, and batch experiments
├── pykt/
│   ├── config/              # Supported model registry
│   ├── datasets/            # Data loaders
│   ├── models/              # Models, EDL, training, and evaluation
│   ├── preprocess/          # Data preprocessing and sequence construction
│   └── utils/               # Shared utilities
├── docs/                    # Documentation and legacy materials
├── tests/                   # Data checks and test scripts
├── setup.py
└── LICENSE
```

Some files under `docs/` retain upstream instructions and may differ from the current model scope or installation procedure. Use this README and the scripts under `examples/` for the current workflow. Retained preprocessing modules do not imply that every model/dataset combination has been validated.

## Acknowledgments and License

This project builds on pyKT, retaining its data processing and knowledge tracing experiment framework while integrating EDL and uncertainty evaluation into the current model set. The upstream project address is recorded in the metadata in `setup.py`.

The code is distributed under the MIT License; see [LICENSE](LICENSE). The original pykt-team copyright notice is retained. When using this project in research, acknowledge pyKT and the specific knowledge tracing models and datasets used in your experiments.
