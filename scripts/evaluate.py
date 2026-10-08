#!/usr/bin/env python3
from __future__ import annotations
"""
Evaluation Pipeline for Speech Emotion Recognition (SER) and Acoustic Classification
=====================================================================================

This script evaluates machine learning models on acoustic and speech representations
for speech emotion recognition across different experimental configurations, datasets,
and validation protocols.

Key Capabilities:
-----------------
1. Feature Representations:
   - Self-Supervised Learning (SSL) Speech Foundations:
     - HuBERT, Whisper, Wav2Vec 2.0, WavLM, FRILL, TRILL, TRILLsson.
   - Handcrafted / Acoustic Representations:
     - eGeMAPS (v02_88), ComParE 2016 (6k), pyAudioAnalysis (pAA), VGGish.

2. Fusion Strategies (FUSION):
   - 'early': Feature-level fusion. Concatenates all feature representations into a single
              unified feature vector per audio sample prior to model training.
   - 'late' : Decision-level fusion. Trains individual models per feature representation
              and exports test predictions (.npy) for downstream soft-voting or ensemble.

3. Evaluation Protocols (MODE):
   - 'inner'          : Standard stratified 80/20 train/test split within the same dataset.
   - 'inner-speaker'  : Speaker-independent split (leave-speakers-out), ensuring disjoint
                        train and test speakers, balanced across female and male speakers.
   - 'inner-sentence' : Sentence-independent split (leave-sentences-out), ensuring lexical
                        content independence between train and test partitions.
   - 'cross'          : Cross-corpus evaluation. Trains on a source dataset and evaluates
                        on an unseen target dataset ('otherData').
   - 'multi'          : Multi-corpus evaluation. Pools multiple datasets together for training
                        and evaluation.

4. Feature Selection / Concatenation (CONCAT):
   - False: Evaluates each feature representation individually (unimodal evaluation).
   - True : Concatenates all features in the configuration into a single vector (early fusion).

5. Optimization and Robustness:
   - Hyperparameter optimization via Stratified 5-Fold GridSearchCV optimizing macro F1-score.
   - Repeated multi-seed evaluation (seeds: [42, 97, 123, 2025]) reporting mean and std dev.
   - Automatic generation and export of confusion matrix PDFs per model and seed.
   - Comprehensive CSV logging with macro precision, recall, F1, balanced accuracy, and best parameters.
"""

import os
import sys
import gc
import random
import warnings
import argparse
import json
from pathlib import Path
from datetime import datetime

try:
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')  # Headless backend for generating figures without display server
    import matplotlib.pyplot as plt

    from scipy import stats
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.base import clone
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn import svm
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.feature_selection import VarianceThreshold
except ImportError as err:
    if '--help' in sys.argv or '-h' in sys.argv:
        np = None
        plt = None
        svm = None
    else:
        raise ImportError(
            f"Required dependency missing: {err}. "
            "Please ensure you have activated your environment (e.g., conda activate admodel) "
            "or installed packages from requirements.txt."
        ) from err


# Optional gradient boosting libraries with graceful fallback
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# ==============================================================================
# Environment and Path Setup
# ==============================================================================
# Disable annoying pydevd warnings related to frozen modules during multiprocessing
os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

# Collect garbage to free unreferenced memory before heavy computations
gc.collect()

# Ensure project root and model directory are accessible in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_ROOT / 'model'

if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import config with fallback to avoid failure if ffmpeg is missing for utils
try:
    from model import config
except Exception:
    import importlib.util
    config_file = MODEL_DIR / 'model' / 'config.py'
    if config_file.exists():
        spec = importlib.util.spec_from_file_location("model.config", config_file)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
    else:
        raise ImportError(f"Could not locate config module in {config_file}")

# Directory for storing generated confusion matrix PDF plots
CM_DIR = Path("confusion_matrices")

# Default emotions for VERBO dataset (7 classes in Portuguese):
# ale: Alegria (Joy)
# des: Desprezo (Contempt)
# med: Medo (Fear)
# neu: Neutro (Neutral)
# rai: Raiva (Anger)
# sur: Surpresa (Surprise)
# tri: Tristeza (Sadness)
DEFAULT_EMOTIONS = ['ale', 'des', 'med', 'neu', 'rai', 'sur', 'tri']


# ==============================================================================
# Global Configuration Variables (Default Settings)
# ==============================================================================
# FUSION: Type of fusion strategy
#   - 'early': Feature-level concatenation across representations
#   - 'late' : Decision-level fusion (saves predictions for soft-voting ensemble)
FUSION = 'late'  # Options: 'early', 'late'

# MODE: Cross-validation and evaluation protocol
#   - 'inner'          : Stratified 80/20 train/test split within the same dataset
#   - 'cross'          : Cross-corpus evaluation (train on source, test on otherData)
#   - 'multi'          : Multi-corpus evaluation (pools multiple datasets)
#   - 'inner-speaker'  : Speaker-independent split (leave-speakers-out)
#   - 'inner-sentence' : Sentence-independent split (leave-sentences-out)
MODE = 'inner'  # Options: 'cross', 'multi', 'inner', 'inner-speaker', 'inner-sentence'

# CONCAT: Feature concatenation toggle
#   - False: Evaluate each feature representation independently (unimodal)
#   - True : Concatenate all features into a single high-dimensional vector (early fusion)
CONCAT = False

# CONF: Dataset and feature representations to evaluate
CONF = [
    {
        "dataset": "verbo",
        "features": [
            'hubert',
            'whisper',
            'wav2vec2',
            'wavlm',
            'frill',
            'eGeMAPSv02_88',
            'pAA',
            'vggish',
            'trillsson5',
            'ComParE_2016_6k',
            'trill',
        ],
    },
]


# ==============================================================================
# Utility & Data Handling Functions
# ==============================================================================

def append_result_file(line: str, filename: str = "eva_late_bah.csv"):
    """
    Append a CSV formatted result line to the results log file.

    If the target file does not exist or is empty, write the standard
    CSV column header first to ensure consistent tabular formatting.

    Args:
        line (str): Comma-separated values formatted result string.
        filename (str): Target CSV file path (defaults to "eva_late_bah.csv").
    """
    output_path = Path(filename)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with open(output_path, 'a', encoding='utf-8') as output_file:
        if write_header:
            header = (
                "dataset,feature,algorithm,balanced_accuracy_mean,balanced_accuracy_std,"
                "precision_macro_mean,precision_macro_std,recall_macro_mean,recall_macro_std,"
                "f1_macro_mean,f1_macro_std,best_params,timestamp"
            )
            output_file.write(header + '\n')
        output_file.write(line + '\n')


def create_labels(class_features_dict: dict):
    """
    Convert a dictionary of per-class feature arrays into combined feature and label matrices.

    Args:
        class_features_dict (dict): Dictionary mapping class index (int) to a 2D numpy
                                    array of feature vectors with shape (n_samples, n_features).

    Returns:
        features (np.ndarray): Combined feature matrix of shape (total_samples, n_features).
        labels (np.ndarray): 1D array of class labels with shape (total_samples,).
    """
    features = []
    labels = []
    for label, feats in class_features_dict.items():
        if feats.size > 0:
            features.append(feats)
            labels.append(np.full(feats.shape[0], label))

    if not features:
        raise ValueError("No feature samples found in class_features_dict.")

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels


def load_features(folder_path: Path | str) -> np.ndarray:
    """
    Load and preprocess feature vectors stored as .npy files within a given directory.

    Each .npy file corresponds to the extracted feature vector for one audio sample.
    The arrays are flattened to 1D, and NaNs are replaced with zero using np.nan_to_num.
    Files containing all NaN values are skipped.

    Args:
        folder_path (Path | str): Path to directory containing .npy feature files.

    Returns:
        np.ndarray: 2D array of shape (n_samples, n_features).
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {folder}")

    features = []
    for filename in sorted(os.listdir(folder)):
        if filename.endswith('.npy'):
            file_path = folder / filename
            data = np.load(file_path)
            # Skip corrupted or entirely invalid feature representations
            if np.isnan(data).all():
                continue
            # Replace individual NaNs with 0 and flatten multi-dimensional vectors
            features.append(np.nan_to_num(data.flatten()))

    if not features:
        return np.empty((0, 0))
    return np.array(features)


def prepare_data(dataset: str, feature: str | list, class_names: list = None):
    """
    Load and assemble feature representations and labels for a dataset.

    Supports both single-feature evaluation and multi-feature early fusion:
    - If `feature` is a single string: Loads features for that specific representation.
    - If `feature` is a list of strings: Loads and horizontally concatenates all features
      along axis 1 for each sample, performing feature-level (early) fusion.

    Args:
        dataset (str): Name of the dataset (e.g., 'verbo').
        feature (str | list): Feature representation name (str) or list of feature names.
        class_names (list, optional): List of class folder names. Defaults to DEFAULT_EMOTIONS.

    Returns:
        features (np.ndarray): Shape (total_samples, total_features).
        labels (np.ndarray): Shape (total_samples,).
    """
    if class_names is None:
        class_names = DEFAULT_EMOTIONS

    class_features_dict = {}
    if isinstance(feature, list):
        # Multi-feature early fusion: Concatenate feature representations column-wise
        for class_idx, class_name in enumerate(class_names):
            first_feat_dir = config.dir_feature.joinpath(feature[0], dataset, class_name)
            feats = load_features(first_feat_dir)
            for feat in feature[1:]:
                next_feat_dir = config.dir_feature.joinpath(feat, dataset, class_name)
                next_feats = load_features(next_feat_dir)
                feats = np.concatenate((feats, next_feats), axis=1)
            class_features_dict[class_idx] = feats
    else:
        # Single feature representation
        for class_idx, class_name in enumerate(class_names):
            feat_dir = config.dir_feature.joinpath(feature, dataset, class_name)
            feats = load_features(feat_dir)
            print(f"[{dataset}] Class {class_name} ({class_idx}): {np.shape(feats)}")
            class_features_dict[class_idx] = feats

    return create_labels(class_features_dict)


def prepare_data_with_speakers(dataset: str, feature: str | list, class_names: list = None):
    """
    Load feature representations and extract speaker IDs from filename conventions.

    Expected filename format: '<emotion>-<speaker>-<sentence>.npy'
    Example: 'ale-f01-s01.npy' -> speaker_id = 'f01'

    Args:
        dataset (str): Name of the dataset (e.g., 'verbo').
        feature (str | list): Single feature name or list of features to concatenate.
        class_names (list, optional): List of class folder names. Defaults to DEFAULT_EMOTIONS.

    Returns:
        features (np.ndarray): Shape (n_samples, n_features).
        labels (np.ndarray): Shape (n_samples,).
        speakers (np.ndarray): Shape (n_samples,) containing string speaker IDs.
    """
    if class_names is None:
        class_names = DEFAULT_EMOTIONS

    feature_names = feature if isinstance(feature, list) else [feature]
    features = []
    labels = []
    speakers = []

    for class_idx, class_name in enumerate(class_names):
        reference_dir = config.dir_feature.joinpath(feature_names[0], dataset, class_name)
        if not reference_dir.exists():
            raise FileNotFoundError(f"Feature directory does not exist: {reference_dir}")

        for filename in sorted(os.listdir(reference_dir)):
            if not filename.endswith('.npy'):
                continue

            name_parts = os.path.splitext(filename)[0].split('-')
            if len(name_parts) < 2:
                raise ValueError(f"Cannot extract speaker ID from feature filename: {filename}")
            speaker_id = name_parts[1].lower()

            feature_vectors = []
            for feature_name in feature_names:
                feature_path = config.dir_feature.joinpath(feature_name, dataset, class_name, filename)
                data = np.load(feature_path)
                if np.isnan(data).all():
                    feature_vectors = []
                    break
                feature_vectors.append(np.nan_to_num(data.flatten()))

            if feature_vectors:
                features.append(np.concatenate(feature_vectors))
                labels.append(class_idx)
                speakers.append(speaker_id)

    return np.asarray(features), np.asarray(labels), np.asarray(speakers)


def split_by_speaker(X: np.ndarray, y: np.ndarray, speakers: np.ndarray, seed: int = 97):
    """
    Perform a speaker-independent train/test partition (leave-speakers-out).

    Partitions samples so that the test set contains exactly one female speaker ('f*')
    and one male speaker ('m*'), ensuring gender balance and avoiding speaker leakage
    between training and testing splits.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        speakers (np.ndarray): Array of speaker IDs corresponding to samples.
        seed (int): Random seed for reproducible speaker selection.

    Returns:
        X_train, X_test, y_train, y_test, test_speakers (list)
    """
    unique_speakers = set(speakers)
    female_speakers = sorted(speaker for speaker in unique_speakers if speaker.startswith('f'))
    male_speakers = sorted(speaker for speaker in unique_speakers if speaker.startswith('m'))

    if not female_speakers or not male_speakers:
        raise ValueError("Speaker split requires at least one female (f*) and one male (m*) speaker.")

    rng = random.Random(seed)
    test_speakers = {rng.choice(female_speakers), rng.choice(male_speakers)}
    test_mask = np.isin(speakers, list(test_speakers))

    if not test_mask.any() or test_mask.all():
        raise ValueError("Speaker split must leave samples in both train and test sets.")

    return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask], sorted(test_speakers)


def prepare_data_with_sentences(dataset: str, feature: str | list, class_names: list = None):
    """
    Load feature representations and extract sentence IDs from filename conventions.

    Expected filename format: '<emotion>-<speaker>-<sentence>.npy'
    Example: 'ale-f01-s01.npy' -> sentence_id = 's01'

    Args:
        dataset (str): Name of the dataset (e.g., 'verbo').
        feature (str | list): Single feature name or list of features to concatenate.
        class_names (list, optional): List of class folder names. Defaults to DEFAULT_EMOTIONS.

    Returns:
        features (np.ndarray): Shape (n_samples, n_features).
        labels (np.ndarray): Shape (n_samples,).
        sentences (np.ndarray): Shape (n_samples,) containing string sentence IDs.
    """
    if class_names is None:
        class_names = DEFAULT_EMOTIONS

    feature_names = feature if isinstance(feature, list) else [feature]
    features = []
    labels = []
    sentences = []

    for class_idx, class_name in enumerate(class_names):
        reference_dir = config.dir_feature.joinpath(feature_names[0], dataset, class_name)
        if not reference_dir.exists():
            raise FileNotFoundError(f"Feature directory does not exist: {reference_dir}")

        for filename in sorted(os.listdir(reference_dir)):
            if not filename.endswith('.npy'):
                continue

            name_parts = os.path.splitext(filename)[0].split('-')
            if len(name_parts) < 3:
                raise ValueError(f"Cannot extract sentence ID from feature filename: {filename}")
            sentence_id = name_parts[2].lower()

            feature_vectors = []
            for feature_name in feature_names:
                feature_path = config.dir_feature.joinpath(feature_name, dataset, class_name, filename)
                data = np.load(feature_path)
                if np.isnan(data).all():
                    feature_vectors = []
                    break
                feature_vectors.append(np.nan_to_num(data.flatten()))

            if feature_vectors:
                features.append(np.concatenate(feature_vectors))
                labels.append(class_idx)
                sentences.append(sentence_id)

    return np.asarray(features), np.asarray(labels), np.asarray(sentences)


def split_by_sentence(X: np.ndarray, y: np.ndarray, sentences: np.ndarray, seed: int = 97):
    """
    Perform a sentence-independent train/test partition (leave-sentences-out).

    Selects 2 unique sentences at random to form the test set, ensuring lexical
    independence and testing the model's generalization across unseen verbal content.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Label vector.
        sentences (np.ndarray): Array of sentence IDs corresponding to samples.
        seed (int): Random seed for reproducible sentence selection.

    Returns:
        X_train, X_test, y_train, y_test, test_sentences (list)
    """
    unique_sentences = sorted(set(sentences))
    if not unique_sentences:
        raise ValueError("Sentence split requires at least one sentence.")

    rng = random.Random(seed)
    test_sentences = set(rng.sample(unique_sentences, min(2, len(unique_sentences))))
    test_mask = np.isin(sentences, list(test_sentences))

    if not test_mask.any() or test_mask.all():
        raise ValueError("Sentence split must leave samples in both train and test sets.")

    return X[~test_mask], X[test_mask], y[~test_mask], y[test_mask], sorted(test_sentences)


# ==============================================================================
# Model Pipeline & Evaluation Functions
# ==============================================================================

def create_pipeline(classifier_model) -> Pipeline:
    """
    Construct a scikit-learn Pipeline with feature standardization and a classifier.

    StandardScaler is fitted strictly on the training partition inside each cross-validation
    fold, ensuring zero data leakage into the validation or testing sets.

    Args:
        classifier_model: An initialized scikit-learn compatible classifier.

    Returns:
        Pipeline: Scikit-learn Pipeline consisting of StandardScaler and the classifier.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier_model),
    ])


def plot_confusion_matrix(y_true, y_pred, model_name: str, rede, seed: int,
                          output_dir: Path | str = None, class_names: list = None) -> Path:
    """
    Generate and save a publication-ready confusion matrix plot as a PDF file.

    Args:
        y_true: True ground truth class labels.
        y_pred: Predicted class labels.
        model_name (str): Identifier name of the machine learning model (e.g., 'SVC').
        rede: Feature representation name (str or list of str).
        seed (int): Evaluation seed number.
        output_dir (Path | str, optional): Directory to save the PDF. Defaults to CM_DIR.
        class_names (list, optional): Emotion label strings. Defaults to DEFAULT_EMOTIONS.

    Returns:
        Path: Full path to the exported PDF plot.
    """
    if output_dir is None:
        output_dir = CM_DIR
    if class_names is None:
        class_names = DEFAULT_EMOTIONS

    if len(y_true) > 0 and isinstance(y_true[0], str):
        labels = class_names
    else:
        labels = list(range(len(class_names)))

    rede_name = '_'.join(rede) if isinstance(rede, list) else str(rede)
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')
    ax.set_title(f"Confusion Matrix - {model_name} | {rede_name} | Seed {seed}")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    out_pdf = output_path / f"{model_name}_{rede_name}_{seed}.pdf"

    fig.tight_layout()
    fig.savefig(out_pdf, format='pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Confusion matrix saved to: {out_pdf}")
    return out_pdf


def evaluate_model_with_gridsearch(pipeline: Pipeline, param_grid: dict,
                                   X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray,
                                   seed: int = 42, model_name: str = None,
                                   feat = None, jobs: int = 15):
    """
    Perform 5-fold cross-validation hyperparameter optimization (GridSearchCV) and test evaluation.

    Optimizes hyperparameters using StratifiedKFold cross-validation on the training set
    with 'f1_macro' as the objective scoring metric. The best fitted estimator then predicts
    on the held-out test partition.

    Args:
        pipeline (Pipeline): Preprocessing and classification pipeline.
        param_grid (dict): Hyperparameter grid dictionary.
        X_train (np.ndarray): Training features.
        y_train (np.ndarray): Training labels.
        X_test (np.ndarray): Testing features.
        y_test (np.ndarray): Testing labels.
        seed (int): Random seed for StratifiedKFold partitioning.
        model_name (str, optional): Name of the model for logging and confusion matrix plot.
        feat: Feature representation name for logging and confusion matrix plot.
        jobs (int): Maximum parallel worker jobs for GridSearchCV.

    Returns:
        report (dict): Detailed classification metrics report dictionary.
        best_params (dict): Optimal hyperparameter combination found.
        best_estimator: Best estimator pipeline refitted on all training data.
        y_pred (np.ndarray): Predicted class labels on the test partition.
        cv_results (dict): Complete cross-validation search results.
    """
    # Reduce parallel jobs for tree ensembles to prevent CPU thread over-subscription
    classifier = pipeline.named_steps['classifier']
    if isinstance(classifier, (RandomForestClassifier, DecisionTreeClassifier)) or \
       (XGBClassifier is not None and isinstance(classifier, XGBClassifier)):
        jobs = min(jobs, 8)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring='f1_macro',  # Macro F1-score across emotion classes
        n_jobs=jobs,
        error_score=0        # Assign score 0 to failing candidate configurations instead of NaN
    )

    # Suppress internal scoring warnings for invalid parameter combinations
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Scoring failed*")
        grid_search.fit(X_train, y_train)

    emotions = DEFAULT_EMOTIONS
    y_pred = grid_search.predict(X_test)
    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(len(emotions)),
        target_names=emotions,
        output_dict=True,
        zero_division=0,
    )
    report['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)

    # Export confusion matrix visualization if model and feature identifiers are supplied
    if model_name is not None and feat is not None:
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
            rede=feat,
            seed=seed,
            class_names=emotions
        )

    return report, grid_search.best_params_, grid_search.best_estimator_, y_pred, grid_search.cv_results_


def evaluate_models(models: dict, method: str, feat: str,
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray,
                    fusion: str = 'late', output_csv: str = "eva_late_bah.csv"):
    """
    Evaluate a dictionary of models on a single train/test split.

    Args:
        models (dict): Dict of {name: (model_instance, param_grid)}.
        method (str): Identifier description of evaluation method / split.
        feat (str): Feature representation name.
        X_train, y_train: Training split arrays.
        X_test, y_test: Testing split arrays.
        fusion (str): Fusion mode ('early' or 'late'). If 'late', saves predictions.
        output_csv (str): Path to CSV results log.
    """
    for name, (model, param_grid) in models.items():
        try:
            print(f"Running evaluation: {method} | Model: {name} | Feature: {feat}")
            pipeline = create_pipeline(model)
            report, best_params, best_estimator, y_pred, all_results = evaluate_model_with_gridsearch(
                pipeline, param_grid, X_train, y_train, X_test, y_test,
                seed=42, model_name=name, feat=feat
            )
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(report)
            macro = report['macro avg']
            append_result_file(
                f"{method},{feat},{name},{report['accuracy']},{macro['precision']},{macro['recall']},{macro['f1-score']},{0},\"{best_params}\",{current_time}",
                filename=output_csv
            )

            # If late fusion, save predictions for soft-voting ensemble
            if fusion == 'late':
                eva_dir = config.dir_processed.joinpath('predicted', "Soft_Voting", method, name)
                os.makedirs(eva_dir, exist_ok=True)
                np.save(eva_dir.joinpath(feat), y_pred)

        except Exception as e:
            print(f"Error evaluating {name} on {feat}: {e}")
            append_result_file(
                f"{method},{feat},{name},0,0,0,0,0,0,\"\",\"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\"",
                filename=output_csv
            )


def evaluate_models_repeated(models: dict, method: str, feat, X: np.ndarray, y: np.ndarray,
                             split_mode: str, groups: np.ndarray = None,
                             X_test: np.ndarray = None, y_test: np.ndarray = None,
                             fusion: str = 'late', seeds: list = None,
                             output_csv: str = "eva_late_bah.csv"):
    """
    Perform repeated evaluation over multiple random seeds, aggregating statistics.

    Executes train/test partitioning and GridSearchCV for each seed in `seeds`.
    Tracks balanced accuracy, macro precision, recall, and F1-score across seeds,
    and computes empirical mean and sample standard deviation (ddof=1).
    When `fusion == 'late'`, saves predictions for each seed to disk for ensemble evaluation.

    Args:
        models (dict): Dict of {name: (model_instance, param_grid)}.
        method (str): Evaluation identifier (e.g., 'verbo', 'verbo-speaker').
        feat: Feature representation name (str or list of str).
        X (np.ndarray): Full dataset feature matrix (or source dataset for 'cross').
        y (np.ndarray): Full dataset label vector (or source dataset for 'cross').
        split_mode (str): 'inner', 'cross', 'multi', 'inner-speaker', or 'inner-sentence'.
        groups (np.ndarray, optional): Speaker or sentence IDs for grouped splits.
        X_test (np.ndarray, optional): Held-out test features (used for 'cross' mode).
        y_test (np.ndarray, optional): Held-out test labels (used for 'cross' mode).
        fusion (str): Fusion mode ('early' or 'late').
        seeds (list, optional): List of random seeds. Defaults to [42, 97, 123, 2025].
        output_csv (str): Output results CSV file path.
    """
    if seeds is None:
        seeds = [42, 97, 123, 2025]

    metric_names = ('balanced_accuracy', 'precision', 'recall', 'f1-score')
    scores = {name: {metric: [] for metric in metric_names} for name in models}
    best_params = {name: [] for name in models}

    for seed in seeds:
        # Partition data according to the selected validation protocol
        if split_mode in ('inner', 'multi'):
            # Standard stratified 80/20 train/test split
            X_train, current_X_test, y_train, current_y_test = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
        elif split_mode == 'inner-speaker':
            # Speaker-independent partition
            X_train, current_X_test, y_train, current_y_test, test_groups = split_by_speaker(
                X, y, groups, seed=seed
            )
            print(f"Seed {seed}: test speakers = {test_groups}")
        elif split_mode == 'inner-sentence':
            # Sentence-independent partition
            X_train, current_X_test, y_train, current_y_test, test_groups = split_by_sentence(
                X, y, groups, seed=seed
            )
            print(f"Seed {seed}: test sentences = {test_groups}")
        elif split_mode == 'cross':
            # Cross-corpus: Train on entire source dataset, test on held-out target dataset
            X_train, y_train = X, y
            current_X_test, current_y_test = X_test, y_test
        else:
            raise ValueError(f"Unsupported evaluation mode: {split_mode}")

        # Evaluate each classifier model on the current seed's partition
        for name, (model, param_grid) in models.items():
            print(f"Evaluating: {method} | Model: {name} | Feat: {feat} | Seed: {seed}")
            seeded_model = clone(model)
            if 'random_state' in seeded_model.get_params(deep=False):
                seeded_model.set_params(random_state=seed)

            pipeline = create_pipeline(seeded_model)
            report, params, _, y_pred, _ = evaluate_model_with_gridsearch(
                pipeline, param_grid, X_train, y_train, current_X_test, current_y_test,
                seed=seed, model_name=name, feat=feat
            )
            macro = report['macro avg']
            scores[name]['balanced_accuracy'].append(report['balanced_accuracy'])
            scores[name]['precision'].append(macro['precision'])
            scores[name]['recall'].append(macro['recall'])
            scores[name]['f1-score'].append(macro['f1-score'])
            best_params[name].append(params)

            # Save predictions for late fusion (decision ensemble / soft voting)
            if fusion == 'late':
                feat_str = '_'.join(feat) if isinstance(feat, list) else str(feat)
                prediction_dir = config.dir_processed.joinpath('predicted', 'Soft_Voting', method, name)
                os.makedirs(prediction_dir, exist_ok=True)
                np.save(prediction_dir.joinpath(f"{feat_str}_seed{seed}.npy"), y_pred)

    # Compute mean and standard deviation across seeds and append to CSV
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for name in models:
        aggregates = {}
        for metric in metric_names:
            values = np.asarray(scores[name][metric], dtype=float)
            aggregates[metric] = (values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0)

        summary_metrics = ", ".join(
            f"{metric}={mean:.4f} +/- {std:.4f}"
            for metric, (mean, std) in aggregates.items()
        )
        print(f"Summary for {method} | {name} | {feat}: {summary_metrics}")

        params_summary = '; '.join(str(params) for params in best_params[name])
        append_result_file(
            f"{method},{feat},{name},"
            f"{aggregates['balanced_accuracy'][0]:.4f},{aggregates['balanced_accuracy'][1]:.4f},"
            f"{aggregates['precision'][0]:.4f},{aggregates['precision'][1]:.4f},"
            f"{aggregates['recall'][0]:.4f},{aggregates['recall'][1]:.4f},"
            f"{aggregates['f1-score'][0]:.4f},{aggregates['f1-score'][1]:.4f},"
            f"\"{params_summary}\",{current_time}",
            filename=output_csv
        )


# ==============================================================================
# Model Definition Dictionary
# ==============================================================================

def get_default_models() -> dict:
    """
    Construct the dictionary of machine learning classifiers and hyperparameter grids.

    The active default model is Support Vector Classifier (SVC). Alternative classifiers
    (Logistic Regression, KNN, MLP, Decision Trees, Random Forest, CatBoost, LightGBM,
    and XGBoost) are configured and can be enabled by uncommenting or selecting them.

    Returns:
        dict: Mapping of model names to (estimator_instance, parameter_grid_dict) tuples.
    """
    models = {
        # --------------------------------------------------------------------------
        # Support Vector Classifier (SVC) - Active Default Model
        # Tested with polynomial kernel, C=100, degree=2, and probability calibration.
        # --------------------------------------------------------------------------
        'SVC': (svm.SVC(random_state=97), {
            'classifier__C': [100],
            'classifier__kernel': ['poly'],                    # Polynomial kernel
            'classifier__gamma': ['scale'],                   # Kernel coefficient
            'classifier__degree': [2],                        # Degree of polynomial kernel
            'classifier__coef0': [0.1],                       # Independent term in kernel function
            'classifier__shrinking': [True],                  # Heuristic shrinking algorithm
            'classifier__probability': [True],                # Enable probability estimates for soft voting
            'classifier__class_weight': [None],               # None or 'balanced'
            'classifier__decision_function_shape': ['ovr'],   # One-vs-rest multiclass strategy
            'classifier__break_ties': [True],                 # Break ties according to decision function
            'classifier__cache_size': [200],                  # Kernel cache size in MB
        }),

        # --- Alternative Full SVC Grid Search Configuration ---
        # 'SVC_full': (svm.SVC(random_state=42), {
        #     'classifier__C': [0.001, 0.01, 0.1, 100] + list(range(1, 11, 3)),
        #     'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
        #     'classifier__gamma': ['scale', 'auto', 0.01, 0.1, 1],
        #     'classifier__degree': [2, 3, 4],
        #     'classifier__coef0': [0.0, 0.1, 0.5],
        #     'classifier__shrinking': [True, False],
        #     'classifier__probability': [True, False],
        #     'classifier__class_weight': [None, 'balanced'],
        #     'classifier__decision_function_shape': ['ovr'],
        #     'classifier__break_ties': [True, False],
        #     'classifier__cache_size': [200],
        # }),

        # --- Logistic Regression (LR) ---
        # 'LR': (LogisticRegression(random_state=42, max_iter=1000), {
        #     'classifier__C': [0.0001, 0.001, 0.01, 0.1, 0.5, 100, 1000] + list(range(1, 10, 2)),
        #     'classifier__penalty': ['l1', 'l2', 'elasticnet', None],
        #     'classifier__solver': ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
        # }),

        # --- K-Nearest Neighbors (KNN) ---
        # 'KNN': (KNeighborsClassifier(), {
        #     'classifier__n_neighbors': list(range(1, 20, 1)),
        #     'classifier__p': [1, 2, 3],
        #     'classifier__weights': ['uniform', 'distance'],
        #     'classifier__metric': ['minkowski', 'euclidean', 'manhattan'],
        # }),

        # --- Multi-Layer Perceptron (MLP) Neural Network ---
        # 'MP': (MLPClassifier(random_state=42), {
        #     'classifier__activation': ['logistic', 'tanh', 'relu'],
        #     'classifier__alpha': [0.0001, 0.001, 0.01],
        #     'classifier__solver': ['adam', 'lbfgs', 'sgd'],
        #     'classifier__max_iter': [300, 500, 1000],
        #     'classifier__hidden_layer_sizes': [
        #         (100,), (200,), (300,),
        #         (100, 100), (200, 100), (300, 150),
        #         (100, 50, 25), (200, 100, 50)
        #     ],
        # }),

        # --- Decision Tree (DT) ---
        # 'DT': (DecisionTreeClassifier(random_state=42), {
        #     'classifier__max_depth': list(range(3, 11)) + [50, None],
        #     'classifier__min_samples_split': [2, 10, 20, 30],
        #     'classifier__min_samples_leaf': [4, 10],
        #     'classifier__max_features': [None, 'sqrt', 'log2'],
        #     'classifier__criterion': ['entropy', 'gini'],
        # }),

        # --- Random Forest (RF) ---
        # 'RF': (RandomForestClassifier(random_state=42), {
        #     'classifier__n_estimators': list(range(50, 300, 30)),
        #     'classifier__max_depth': list(range(3, 11)) + [30, 50, 100, None],
        #     'classifier__criterion': ['entropy', 'gini'],
        # }),
    }

    # Optional gradient boosting models (only added if libraries are installed)
    if CatBoostClassifier is not None:
        pass
        # 'CatBoost': (CatBoostClassifier(random_state=42, task_type="CPU", verbose=0), {
        #     'classifier__iterations': [70, 100, 200],
        #     'classifier__depth': [3, 6],
        #     'classifier__learning_rate': [0.01, 0.05, 0.1],
        # })

    if XGBClassifier is not None:
        pass
        # 'XGBoost': (XGBClassifier(random_state=42, verbosity=0), {
        #     'classifier__n_estimators': [70, 100, 200],
        #     'classifier__max_depth': [3, 6],
        #     'classifier__learning_rate': [0.01, 0.05],
        # })

    if lgb is not None:
        pass
        # 'LightGBM': (lgb.LGBMClassifier(random_state=42, verbose=-1), {
        #     'classifier__n_estimators': [70, 100, 200],
        #     'classifier__max_depth': [3, 5, 8],
        #     'classifier__learning_rate': [0.01, 0.05],
        # })

    return models


# ==============================================================================
# Pipeline Execution & CLI Argument Parsing
# ==============================================================================

def run_pipeline(conf: list = None, mode: str = MODE, fusion: str = FUSION,
                 concat: bool = CONCAT, output_csv: str = "eva_late_bah.csv",
                 seeds: list = None, models: dict = None):
    """
    Main evaluation workflow iterating over datasets and feature configurations.

    Args:
        conf (list, optional): List of dataset configuration dictionaries.
        mode (str): Evaluation mode ('inner', 'cross', 'multi', 'inner-speaker', 'inner-sentence').
        fusion (str): Fusion strategy ('early' or 'late').
        concat (bool): Whether to concatenate all features into a single representation.
        output_csv (str): Output CSV path for logging results.
        seeds (list, optional): List of random seeds for repeated evaluation.
        models (dict, optional): Model dictionary. Defaults to get_default_models().
    """
    if conf is None:
        conf = CONF
    if models is None:
        models = get_default_models()
    if seeds is None:
        seeds = [42, 97, 123, 2025]

    print("=" * 80)
    print("STARTING EVALUATION PIPELINE")
    print(f"  FUSION     : {fusion}")
    print(f"  MODE       : {mode}")
    print(f"  CONCAT     : {concat}")
    print(f"  OUTPUT CSV : {output_csv}")
    print(f"  SEEDS      : {seeds}")
    print(f"  MODELS     : {list(models.keys())}")
    print("=" * 80)

    for d in conf:
        dataset_name = d.get('dataset', 'unknown')
        features_list = d.get('features', [])
        other_data = d.get('otherData', None)

        try:
            if concat:
                # Early fusion: Concatenate all feature representations into one combined matrix
                feat_identifier = 'all'
                method = ''
                groups = None
                X_test = None
                y_test = None

                if mode == 'inner':
                    X, y = prepare_data(dataset_name, features_list)
                    method = dataset_name

                elif mode == 'inner-speaker':
                    X, y, groups = prepare_data_with_speakers(dataset_name, features_list)
                    method = f"{dataset_name}-speaker"

                elif mode == 'inner-sentence':
                    X, y, groups = prepare_data_with_sentences(dataset_name, features_list)
                    method = f"{dataset_name}-sentence"

                elif mode == 'cross':
                    if not other_data:
                        raise ValueError(f"Cross-corpus evaluation requires 'otherData' in configuration: {d}")
                    method = f"{dataset_name} -> {other_data}"
                    # Train on source dataset, test on target dataset
                    X_train, y_train = prepare_data(dataset_name, features_list)
                    X_test, y_test = prepare_data(other_data, features_list)
                    X, y = X_train, y_train

                elif mode == 'multi':
                    # Multi-corpus: Pool all datasets in conf together
                    method = "multi-" + "_".join([item['dataset'] for item in conf])
                    all_X, all_y = [], []
                    for item in conf:
                        x_item, y_item = prepare_data(item['dataset'], features_list)
                        all_X.append(x_item)
                        all_y.append(y_item)
                    X = np.concatenate(all_X, axis=0)
                    y = np.concatenate(all_y, axis=0)

                else:
                    raise ValueError(f"Unsupported evaluation mode: {mode}")

                evaluate_models_repeated(
                    models=models,
                    method=method,
                    feat=feat_identifier,
                    X=X,
                    y=y,
                    split_mode=mode,
                    groups=groups,
                    X_test=X_test,
                    y_test=y_test,
                    fusion=fusion,
                    seeds=seeds,
                    output_csv=output_csv
                )

            else:
                # Unimodal evaluation: Evaluate each feature representation independently
                for feat in features_list:
                    method = ''
                    groups = None
                    X_test = None
                    y_test = None

                    if mode == 'inner':
                        X, y = prepare_data(dataset_name, feat)
                        method = dataset_name

                    elif mode == 'inner-speaker':
                        X, y, groups = prepare_data_with_speakers(dataset_name, feat)
                        method = f"{dataset_name}-speaker"

                    elif mode == 'inner-sentence':
                        X, y, groups = prepare_data_with_sentences(dataset_name, feat)
                        method = f"{dataset_name}-sentence"

                    elif mode == 'cross':
                        if not other_data:
                            raise ValueError(f"Cross-corpus evaluation requires 'otherData' in configuration: {d}")
                        method = f"{dataset_name} -> {other_data}"
                        X_train, y_train = prepare_data(dataset_name, feat)
                        X_test, y_test = prepare_data(other_data, feat)
                        X, y = X_train, y_train

                    elif mode == 'multi':
                        method = "multi-" + "_".join([item['dataset'] for item in conf])
                        all_X, all_y = [], []
                        for item in conf:
                            x_item, y_item = prepare_data(item['dataset'], feat)
                            all_X.append(x_item)
                            all_y.append(y_item)
                        X = np.concatenate(all_X, axis=0)
                        y = np.concatenate(all_y, axis=0)

                    else:
                        raise ValueError(f"Unsupported evaluation mode: {mode}")

                    evaluate_models_repeated(
                        models=models,
                        method=method,
                        feat=feat,
                        X=X,
                        y=y,
                        split_mode=mode,
                        groups=groups,
                        X_test=X_test,
                        y_test=y_test,
                        fusion=fusion,
                        seeds=seeds,
                        output_csv=output_csv
                    )

        except Exception as e:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Error during evaluation loop for dataset {dataset_name}: {e}")
            append_result_file(
                f"{dataset_name},all,{e},0,0,0,0,0,0,\"\",\"{current_time}\"",
                filename=output_csv
            )


def parse_arguments():
    """
    Parse command-line arguments for pipeline configuration.

    Allows overriding FUSION, MODE, CONCAT, and CONF variables from the command line
    or shell scripts like `run.sh`. If arguments are omitted, fallback to the default
    values specified in the script.

    Returns:
        argparse.Namespace: Parsed CLI options.
    """
    parser = argparse.ArgumentParser(
        description="Speech Emotion Recognition & Acoustic Classification Evaluation Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--fusion',
        type=str,
        default=FUSION,
        choices=['early', 'late'],
        help="Fusion strategy: 'early' for feature concatenation, 'late' for decision-level voting"
    )

    parser.add_argument(
        '--mode',
        type=str,
        default=MODE,
        choices=['inner', 'cross', 'multi', 'inner-speaker', 'inner-sentence'],
        help="Evaluation protocol / splitting mode"
    )

    parser.add_argument(
        '--concat',
        action='store_true',
        default=CONCAT,
        help="Concatenate all features into a single feature vector (early fusion)"
    )

    parser.add_argument(
        '--no-concat',
        action='store_false',
        dest='concat',
        help="Evaluate each feature representation individually (unimodal)"
    )

    parser.add_argument(
        '--dataset',
        type=str,
        default="verbo",
        help="Primary dataset name to evaluate"
    )

    parser.add_argument(
        '--other-data',
        type=str,
        default=None,
        help="Target dataset name for cross-corpus evaluation (used when --mode cross)"
    )

    parser.add_argument(
        '--features',
        type=str,
        nargs='+',
        default=[
            'hubert', 'whisper', 'wav2vec2', 'wavlm', 'frill',
            'eGeMAPSv02_88', 'pAA', 'vggish', 'trillsson5',
            'ComParE_2016_6k', 'trill'
        ],
        help="List of acoustic/SSL feature representations to evaluate"
    )

    parser.add_argument(
        '--conf-json',
        type=str,
        default=None,
        help="Optional JSON string or path to JSON file specifying full CONF configuration list"
    )

    parser.add_argument(
        '--output-csv',
        type=str,
        default="eva_late_bah.csv",
        help="Output CSV file path to append evaluation results"
    )

    parser.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=[42, 97, 123, 2025],
        help="List of random seeds for repeated evaluation"
    )

    return parser.parse_args()


# ==============================================================================
# Script Entry Point
# ==============================================================================

if __name__ == '__main__':
    args = parse_arguments()

    # Determine CONF structure:
    # 1. From JSON file or raw JSON string if provided
    # 2. Otherwise constructed from --dataset, --other-data, and --features
    if args.conf_json:
        if os.path.exists(args.conf_json):
            with open(args.conf_json, 'r', encoding='utf-8') as f:
                active_conf = json.load(f)
        else:
            active_conf = json.loads(args.conf_json)
    else:
        conf_entry = {
            "dataset": args.dataset,
            "features": args.features,
        }
        if args.other_data:
            conf_entry["otherData"] = args.other_data
        active_conf = [conf_entry]

    # Run the evaluation pipeline
    run_pipeline(
        conf=active_conf,
        mode=args.mode,
        fusion=args.fusion,
        concat=args.concat,
        output_csv=args.output_csv,
        seeds=args.seeds,
        models=get_default_models()
    )