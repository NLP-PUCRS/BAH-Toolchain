# BAH-Toolchain

Codebase, models, or supplementary results from the paper "The BAH Toolchain: Class-Aware Speech Emotion Recognition in Portuguese" published at Journal on Interactive Systems.

---

## Table of Contents
- [Project Architecture & Directory Layout](#project-architecture--directory-layout)
- [Environment Setup](#environment-setup)
  - [Step 1: Create & Activate Conda Environment](#step-1-create--activate-conda-environment)
  - [Step 2: Core Audio & System Dependencies](#step-2-core-audio--system-dependencies)
  - [Step 3: PyTorch (Required for SSL Feature Extraction)](#step-3-pytorch-required-for-ssl-feature-extraction)
  - [Step 4: TensorFlow & TF-Hub (Required for Acoustic/TF Embeddings)](#step-4-tensorflow--tf-hub-required-for-acoustictf-embeddings)
  - [Step 5: Install Python Dependencies](#step-5-install-python-dependencies)
  - [Step 6: Install Local `model` Package](#step-6-install-local-model-package)
  - [Optional Acceleration Packages](#optional-acceleration-packages)
- [Step-by-Step Necessity & Import Assessment](#step-by-step-necessity--import-assessment)
- [Dataset Pre-Processing (Mandatory Step)](#dataset-pre-processing-mandatory-step)
- [Running the Evaluation Pipeline](#running-the-evaluation-pipeline)
  - [Configuration Options](#configuration-options)
  - [Execution via `run.sh`](#execution-via-runsh)
  - [Direct Execution via `evaluate.py`](#direct-execution-via-evaluatepy)
- [Outputs & Results](#outputs--results)
- [Suplementary Material from JIS Paper](#suplementary-material-from-jis-paper)
  - [Intra-Corpus Performance Metrics](#intra-corpus-performance-metrics)
  - [Speaker-Independent Performance Metrics](#speaker-independent-performance-metrics)
  - [Visualizations & Plots](#visualizations--plots)
  - [Citation](#citation)

---

## Project Architecture & Directory Layout

To understand the directory structure and data workflow, refer to [data/README.md](data/README.md).

```
BAH-Toolchain/
├── data/
│   ├── Audios/                  # Place unorganized input audio files here
│   ├── raw/                     # Audio files organized by emotion class (e.g. verbo/)
│   ├── features/                # Pre-extracted .npy feature representations
│   └── processed/               # Dataset metadata CSVs and prediction arrays
├── model/
│   ├── model/                   # Core modules (config.py, utils.py)
│   └── setup.py                 # Package setup configuration
├── scripts/
│   ├── pre_processing_VERBO.py  # Dataset organization and metadata extraction script
│   ├── evaluate.py              # ML evaluation pipeline (SVM, trees, boosting)
│   └── run.sh -> ../run.sh      # Symlink to runner script
├── run.sh                       # Configurable bash script to launch evaluate.py
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Environment Setup

### Step 1: Create & Activate Conda Environment
Create a clean Python 3.10 environment and activate it:
```bash
conda create --name bah python=3.10 -y
conda activate bah
```

### Step 2: Core Audio & System Dependencies
Install system-level audio dependencies (PortAudio for `pyaudio`/`pyAudioAnalysis`, and FFmpeg for audio decoding via `ffmpeg-python`):
```bash
# Install portaudio via conda-forge
conda install -c conda-forge portaudio -y

# Ensure FFmpeg is installed on your system (Linux Ubuntu/Debian):
sudo apt update && sudo apt install -y ffmpeg
```

### Step 3: PyTorch (Required for SSL Feature Extraction)
Install PyTorch for deep learning speech foundations (HuBERT, Wav2Vec 2.0, WavLM, Whisper, PyAnnote):
```bash
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1+:
# pip install torch torchvision torchaudio

# For CPU-only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Step 4: TensorFlow & TF-Hub (Required for Acoustic/TF Embeddings)
Install TensorFlow Hub for pretrained acoustic embeddings (TRILL, FRILL, VGGish):
```bash
pip install --upgrade tensorflow-hub
```

### Step 5: Install Python Dependencies
Install all library requirements from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 6: Install Local `model` Package
Install the project's local `model` package in editable mode so that `from model import config` is accessible anywhere:
```bash
pip install -e model
```

### Optional Acceleration Packages
These packages are **NOT required** for running pre-processing or the evaluation pipeline:

* **FlashAttention (`flash-attn`)**: *Optional*. Only used for accelerated self-attention training with specific transformer architectures. Requires GPU with matching CUDA toolkit (`nvcc`). Skip if evaluating pre-extracted features:
  ```bash
  # Optional:
  pip install flash-attn --no-build-isolation
  ```
* **LightGBM CUDA build from source**: *Optional*. `requirements.txt` already provides CPU LightGBM (`lightgbm==4.5.0`). Compiling from source with GPU is only needed if you specifically train LightGBM on GPU:
  ```bash
  # Optional (GPU LightGBM build):
  git clone --recursive https://github.com/microsoft/LightGBM
  cd LightGBM && cmake -B build -S . -DUSE_GPU=1 && sudo sh ./build-python.sh install --precompile && cd ..
  ```

---

## Step-by-Step Necessity & Import Assessment

| Setup Step / Package | Used By | Necessity | Explanation |
| :--- | :--- | :--- | :--- |
| **`conda activate bah`** | Entire project | **Mandatory** | Isolates Python 3.10 dependencies. |
| **`pip install -r requirements.txt`** | All scripts | **Mandatory** | Installs `scikit-learn`, `numpy`, `pandas`, `scipy`, `catboost`, `xgboost`, audio libraries. |
| **`pip install -e model`** | `evaluate.py`, `utils.py` | **Mandatory** | Exposes `model.config` and `model.utils`. *(Replaces legacy broken `calpy-master` step).* |
| **`portaudio` & `ffmpeg`** | `model/model/utils.py` | **Mandatory** | Required for `load_audio` via `ffmpeg-python` and audio I/O with `pyAudioAnalysis`. |
| **PyTorch (`torch`)** | Feature extraction | **Required for SSL** | Required to extract SSL features (HuBERT, Whisper, Wav2Vec 2.0, WavLM). |
| **TensorFlow & TF-Hub** | Feature extraction | **Required for TF** | Required to extract TRILL, FRILL, and VGGish embeddings. |
| **FlashAttention (`flash-attn`)** | None directly in repo | **Optional / Not needed** | Not imported by `evaluate.py` or `pre_processing_VERBO.py`. Skip to avoid build failures. |
| **LightGBM CUDA from source** | `evaluate.py` | **Optional / Not needed** | `lightgbm==4.5.0` is already in `requirements.txt` and runs on CPU. GPU build is optional. |

---

## Dataset Pre-Processing (Mandatory Step)

> [!IMPORTANT]
> **After configuring the environment, the dataset must be pre-processed with `pre_processing_VERBO.py` before feature extraction and evaluation can take place.**

The VERBO dataset audio files must be structured into class-specific folders (`data/raw/verbo/<emotion>/`) for feature extraction and speaker/sentence parsing.

### Workflow:
1. **Place raw audio files in `data/Audios/`**:
   Expected structure:
   ```
   data/Audios/
   ├── f01/
   │   ├── ale-f01-s01.wav
   │   ├── des-f01-s01.wav
   │   └── ...
   └── m01/
       ├── ale-m01-s01.wav
       └── ...
   ```
   *Filename format: `<emotion>-<speaker_id>-<sentence_id>.wav`*
   *Emotion labels (7 classes): `ale` (Alegria), `des` (Desprezo), `med` (Medo), `neu` (Neutro), `rai` (Raiva), `sur` (Surpresa), `tri` (Tristeza).*

2. **Execute the pre-processing script**:
   ```bash
   python scripts/pre_processing_VERBO.py
   ```
   *Optional flag:* `--move` to move files instead of copying them (default copies to preserve your original recordings).

3. **Generated outputs**:
   * Organized raw audio: `data/raw/verbo/<emotion>/<audio_file>.wav`
   * Metadata catalogue: `data/processed/verbo.csv`

---

## Running the Evaluation Pipeline

Once features are extracted into `data/features/<feature>/<dataset>/<emotion>/`, you can run the evaluation pipeline using either `run.sh` or `scripts/evaluate.py`.

### Configuration Options

The pipeline supports the following experimental variables:

* **`FUSION`**:
  * `'late'` *(default)*: Decision-level fusion. Evaluates individual feature models and exports prediction arrays (`.npy`) into `data/processed/predicted/Soft_Voting/` for ensemble/soft voting.
  * `'early'`: Feature-level fusion. Concatenates all feature representations into a single unified matrix per sample before training.
* **`MODE`**:
  * `'inner'` *(default)*: Stratified 80/20 train/test split within the dataset.
  * `'inner-speaker'`: Speaker-independent partition (leave-speakers-out, balanced across female and male speakers).
  * `'inner-sentence'`: Sentence-independent partition (leave-sentences-out).
  * `'cross'`: Cross-corpus evaluation (trains on `dataset`, tests on `otherData`).
  * `'multi'`: Multi-corpus evaluation (pools multiple datasets together).
* **`CONCAT`**:
  * `False` *(default)*: Evaluates each feature representation individually (unimodal).
  * `True`: Horizontally concatenates all features in `CONF` into a combined feature vector (early fusion).
* **`CONF`**:
  * Dataset name (`verbo`) and the list of feature representations to evaluate:
    ```python
    CONF = [
        {
            "dataset": "verbo",
            "features": [
                'hubert', 'whisper', 'wav2vec2', 'wavlm', 'frill',
                'eGeMAPSv02_88', 'pAA', 'vggish', 'trillsson5',
                'ComParE_2016_6k', 'trill'
            ],
        },
    ]
    ```

### Execution via `run.sh`

`run.sh` provides an easy-to-use wrapper that auto-detects your conda environment (`bah` or `admodel`):

```bash
# 1. Run with configured defaults (FUSION=late, MODE=inner, CONCAT=False):
./run.sh

# 2. Early fusion with feature concatenation:
./run.sh --fusion early --concat true --mode inner

# 3. Speaker-independent evaluation:
./run.sh --mode inner-speaker

# 4. Cross-corpus evaluation (train on verbo, test on otherData):
./run.sh --mode cross --other-data crema-d

# 5. Show all available options:
./run.sh --help
```

You can also edit the variables directly at the top of `run.sh`.

### Direct Execution via `evaluate.py`

You can also invoke `scripts/evaluate.py` directly:
```bash
# Run with default settings:
python scripts/evaluate.py

# Run with custom parameters:
python scripts/evaluate.py --fusion late --mode inner --no-concat --dataset verbo
```

---

## Outputs & Results

1. **Results Log (`eva_late_bah.csv`)**:
   Appends performance metrics across repeated seeds (`[42, 97, 123, 2025]`):
   * Balanced Accuracy (Mean & Std)
   * Precision Macro (Mean & Std)
   * Recall Macro (Mean & Std)
   * F1 Macro (Mean & Std)
   * Best Hyperparameters selected by GridSearchCV
   * Timestamp
2. **Confusion Matrix Plots (`confusion_matrices/*.pdf`)**:
   Visualizes classification performance per model, feature representation, and seed.
3. **Late Fusion Predictions (`data/processed/predicted/Soft_Voting/`)**:
   Saved prediction `.npy` files for decision-level ensembling.


---

## Suplementary material from JIS paper

This section presents the supplementary evaluation tables and artifacts from the JIS paper, available in the [`jis-results/`](jis-results/) directory.

### Intra-Corpus Performance Metrics

*Table: Performance metrics of different feature extractors using SVC on the Verbo dataset (averaged across seeds 42, 97, 123, 2025).*

| Model | Balanced Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| HuBERT | $0.2984 \pm 0.0170$ | $0.2987 \pm 0.0192$ | $0.2984 \pm 0.0170$ | $0.2969 \pm 0.0180$ |
| Whisper | $0.5545 \pm 0.0228$ | $0.5586 \pm 0.0264$ | $0.5545 \pm 0.0228$ | $0.5538 \pm 0.0242$ |
| wav2vec2 | $0.2984 \pm 0.0170$ | $0.2987 \pm 0.0192$ | $0.2984 \pm 0.0170$ | $0.2969 \pm 0.0180$ |
| WavLM | $0.3664 \pm 0.0427$ | $0.3666 \pm 0.0458$ | $0.3664 \pm 0.0427$ | $0.3631 \pm 0.0452$ |
| FRILL | $0.9008 \pm 0.0287$ | $0.9041 \pm 0.0274$ | $0.9008 \pm 0.0287$ | $0.9000 \pm 0.0288$ |
| eGeMAPSv02_88 | $0.6159 \pm 0.0495$ | $0.6193 \pm 0.0468$ | $0.6159 \pm 0.0495$ | $0.6139 \pm 0.0491$ |
| pAA | $0.5832 \pm 0.0337$ | $0.5917 \pm 0.0380$ | $0.5832 \pm 0.0337$ | $0.5829 \pm 0.0338$ |
| VGGish | $0.6001 \pm 0.0258$ | $0.6075 \pm 0.0312$ | $0.6001 \pm 0.0258$ | $0.6011 \pm 0.0268$ |
| Trillsson5 | $0.7096 \pm 0.0188$ | $0.7132 \pm 0.0192$ | $0.7096 \pm 0.0188$ | $0.7090 \pm 0.0194$ |
| ComParE_2016_6k | $0.5687 \pm 0.0124$ | $0.5738 \pm 0.0099$ | $0.5687 \pm 0.0124$ | $0.5671 \pm 0.0127$ |
| TRILL | $0.9093 \pm 0.0140$ | $0.9103 \pm 0.0141$ | $0.9093 \pm 0.0140$ | $0.9088 \pm 0.0135$ |

---

### Speaker-Independent Performance Metrics

*Table: Performance metrics of different feature extractors using SVC on the Verbo dataset in a speaker-independent setting (averaged across seeds 42, 97, 123, 2025). The speakers on the test set for each seed were: 42 ['f6', 'm1'], 97 ['f2', 'm4'], 123 ['f1', 'm3'], 2025 ['f5', 'm1'].*

| Model | Balanced Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| HuBERT | $0.3380 \pm 0.0866$ | $0.3512 \pm 0.0841$ | $0.3380 \pm 0.0866$ | $0.3308 \pm 0.0881$ |
| Whisper | $0.4490 \pm 0.0896$ | $0.4972 \pm 0.0873$ | $0.4490 \pm 0.0896$ | $0.4460 \pm 0.0928$ |
| wav2vec2 | $0.3380 \pm 0.0866$ | $0.3512 \pm 0.0841$ | $0.3380 \pm 0.0866$ | $0.3308 \pm 0.0881$ |
| WavLM | $0.2436 \pm 0.0307$ | $0.2446 \pm 0.0681$ | $0.2436 \pm 0.0307$ | $0.2259 \pm 0.0391$ |
| FRILL | $0.3571 \pm 0.0515$ | $0.4271 \pm 0.0818$ | $0.3571 \pm 0.0515$ | $0.3495 \pm 0.0592$ |
| eGeMAPSv02_88 | $0.3010 \pm 0.0325$ | $0.3297 \pm 0.0617$ | $0.3010 \pm 0.0325$ | $0.2879 \pm 0.0388$ |
| pAA | $0.3112 \pm 0.0480$ | $0.3367 \pm 0.0267$ | $0.3112 \pm 0.0480$ | $0.3039 \pm 0.0443$ |
| VGGish | $0.3240 \pm 0.0465$ | $0.3448 \pm 0.0804$ | $0.3240 \pm 0.0465$ | $0.3134 \pm 0.0493$ |
| Trillsson5 | $0.6059 \pm 0.0722$ | $0.6571 \pm 0.0768$ | $0.6059 \pm 0.0722$ | $0.5964 \pm 0.0832$ |
| ComParE_2016_6k | $0.3380 \pm 0.1219$ | $0.4302 \pm 0.0874$ | $0.3380 \pm 0.1219$ | $0.3270 \pm 0.1224$ |
| TRILL | $0.3520 \pm 0.0279$ | $0.3646 \pm 0.0107$ | $0.3520 \pm 0.0279$ | $0.3261 \pm 0.0182$ |

---

### Visualizations & Plots

The [`jis-results/`](jis-results/) directory also includes the generated evaluation plots:
1. **Confusion Matrices (`jis-results/confusion_matrices/`)**:
   Individual confusion matrix PDF plots per model, feature representation, and random seed (`42`, `97`, `123`, `2025`).
2. **PCA Ellipses Plots (`jis-results/results_audio/pca/`)**:
   2D PCA projection plots with confidence ellipses showing feature separation across emotion classes:
   * `pca_ellipses_frill.pdf` (FRILL embeddings)
   * `pca_ellipses_trill.pdf` (TRILL embeddings)
   * `pca_ellipses_trillsson5.pdf` (TRILLsson5 embeddings)

```
jis-results/
├── complete_tables/
│   ├── performance_metrics_table.md
│   └── speaker_independent_performance_metrics.md
├── confusion_matrices/
│   ├── SVC_ComParE_2016_6k_42.pdf
│   ├── SVC_frill_42.pdf
│   └── ...
└── results_audio/
    └── pca/
        ├── pca_ellipses_frill.pdf
        ├── pca_ellipses_trill.pdf
        └── pca_ellipses_trillsson5.pdf
```

---

### Citation

If you use this codebase, models, or supplementary results in your research, please cite our paper:

```bibtex
@article{bah-toolchain,
  author    = {Gabriela Federhen de Carvalho and Larissa Guder and Rodrigo Rafael Villarreal Goular and Felipe Meneguzzi and Dalvan Griebler},
  title     = {The BAH Toolchain: Class-Aware Speech Emotion Recognition in Portuguese},
  journal   = {Journal on Interactive Systems (JIS)},
  year      = {2026},
  volume    = {},
  number    = {},
  pages     = {10},
  doi       = {10.5753/jis.2026.XXXX}
}

@inproceedings{webmedia,
 author = {Larissa Guder and Luan Dopke and Marcos Kaiser and Dalvan Griebler and Felipe Meneguzzi},
 title = { BAH: Beyond Acoustic Handcrafted features for speech emotion recognition in Portuguese},
 booktitle = {Proceedings of the 31st Brazilian Symposium on Multimedia and the Web},
 location = {Rio de Janeiro/RJ},
 year = {2025},
 keywords = {},
 issn = {0000-0000},
 pages = {86--93},
 publisher = {SBC},
 address = {Porto Alegre, RS, Brasil},
 doi = {10.5753/webmedia.2025.16129},
 url = {https://sol.sbc.org.br/index.php/webmedia/article/view/37950}
}

```
