#!/usr/bin/env python3
"""
VERBO Dataset Pre-Processing Script
===================================

This script organizes raw audio recordings from the VERBO (Voz, Emoção, Recursos
Biológicos e Operações) dataset into class-specific subdirectories under `data/raw/verbo/`
for downstream acoustic and SSL feature extraction.

Dataset Structure & Parsing:
----------------------------
Expected input folder:
    data/Audios/
    ├── f01/
    │   ├── ale-f01-s01.wav
    │   ├── des-f01-s01.wav
    │   └── ...
    └── m01/
        ├── ale-m01-s01.wav
        └── ...

Filename Convention:
    `<emotion>-<speaker_id>-<sentence_id>.wav`
    Example: `ale-f01-s01.wav`
    - `ale` : Emotion label (ale, des, med, neu, rai, sur, tri)
    - `f01` : Speaker ID (female speaker 01)
    - `s01` : Sentence ID (sentence 01)

Processed Outputs:
------------------
1. Audio files organized by emotion class:
   `data/raw/verbo/<emotion>/<filename>`
   e.g., `data/raw/verbo/ale/ale-f01-s01.wav`

2. Metadata CSV file:
   `data/processed/verbo.csv`
   Columns: path, dataset, arquivo, pessoa, genero, emocao
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

try:
    import pandas as pd
except ImportError as err:
    if '--help' in sys.argv or '-h' in sys.argv:
        pd = None
    else:
        raise ImportError(
            f"Missing required dependency: {err}. "
            "Please ensure you have activated your environment (e.g., conda activate bah) "
            "or installed requirements via 'pip install -r requirements.txt'."
        ) from err

# Determine project directories dynamically
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / 'data' / 'Audios'
DEFAULT_OUTPUT_RAW = PROJECT_ROOT / 'data' / 'raw' / 'verbo'
DEFAULT_METADATA_CSV = PROJECT_ROOT / 'data' / 'processed' / 'verbo.csv'


def parse_path_verbo(path_reg: str) -> tuple:
    """
    Parse metadata from a VERBO audio file path.

    Extracts:
    - pessoa (speaker folder, e.g. 'f01')
    - genero (gender: 'f' if starting with 'f', else 'm')
    - arquivo (audio filename, e.g. 'ale-f01-s01.wav')
    - emocao (emotion label, e.g. 'ale')

    Args:
        path_reg (str): Path to audio file.

    Returns:
        tuple: (path_reg, 'verbo', arquivo, pessoa, genero, emocao)
    """
    path_obj = Path(path_reg)
    arquivo = path_obj.name
    pessoa = path_obj.parent.name
    genero = 'f' if pessoa.lower().startswith('f') else 'm'

    # Extract emotion from filename (prefix before the first hyphen)
    parts = arquivo.split('-')
    emocao = parts[0].lower() if len(parts) > 1 else 'unknown'

    return path_reg, 'verbo', arquivo, pessoa, genero, emocao


def preprocess_verbo(input_dir: Path, output_raw_dir: Path,
                     output_csv_path: Path, move_files: bool = False):
    """
    Scan source directory, extract metadata, create CSV, and organize audio files into emotion classes.

    Args:
        input_dir (Path): Source folder containing unorganized audio files (data/Audios).
        output_raw_dir (Path): Destination directory for organized class folders (data/raw/verbo).
        output_csv_path (Path): Path to output metadata CSV file (data/processed/verbo.csv).
        move_files (bool): If True, move files. If False, copy files to preserve originals.
    """
    if not input_dir.exists():
        print(f"[ERROR] Source audio directory does not exist: {input_dir}")
        print("Please place the VERBO audio files in 'data/Audios/' before running this script.")
        print("Expected structure: data/Audios/<speaker_folder>/<audio_files.wav>")
        sys.exit(1)

    # Collect all audio files (.wav, .mp3, .ogg, etc.)
    audio_extensions = {'.wav', '.mp3', '.ogg', '.flac'}
    audio_files = []
    for root, _, files in os.walk(input_dir):
        for file in sorted(files):
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))

    if not audio_files:
        print(f"[WARNING] No audio files found in {input_dir} with extensions {audio_extensions}.")
        print("Please check that the audio files are placed inside data/Audios/.")
        sys.exit(1)

    print(f"[INFO] Found {len(audio_files)} audio files in {input_dir}.")

    # Build DataFrame with metadata
    data = [parse_path_verbo(f) for f in audio_files]
    df_verbo = pd.DataFrame(
        data=data,
        columns=['path', 'dataset', 'arquivo', 'pessoa', 'genero', 'emocao']
    )

    print("[INFO] Sample extracted metadata:")
    print(df_verbo.head())
    print("\n[INFO] Distribution by emotion:")
    print(df_verbo['emocao'].value_counts())

    # Ensure processed directory exists and save metadata CSV
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_verbo.to_csv(output_csv_path, index=False)
    print(f"\n[INFO] Metadata CSV saved to: {output_csv_path}")

    # Organize audio files into class subdirectories: data/raw/verbo/<emocao>/<arquivo>
    count = 0
    action_str = "Moving" if move_files else "Copying"
    print(f"[INFO] {action_str} audio files to {output_raw_dir} by emotion class...")

    for row in df_verbo.itertuples(index=False):
        src = row.path
        emocao = row.emocao
        dest_dir = output_raw_dir / emocao
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / row.arquivo

        if os.path.exists(src):
            if move_files:
                shutil.move(src, dest_file)
            else:
                shutil.copy2(src, dest_file)
            count += 1

    print(f"[SUCCESS] Successfully processed {count} audio files into {output_raw_dir}.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pre-process VERBO dataset and organize audio files by emotion class.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Source directory containing unorganized VERBO audio files"
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_RAW,
        help="Destination directory for class-organized raw audio files"
    )
    parser.add_argument(
        '--metadata-csv',
        type=Path,
        default=DEFAULT_METADATA_CSV,
        help="Destination path for output metadata CSV"
    )
    parser.add_argument(
        '--move',
        action='store_true',
        help="Move audio files instead of copying them (default: copy to preserve original files)"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    preprocess_verbo(
        input_dir=args.input_dir,
        output_raw_dir=args.output_dir,
        output_csv_path=args.metadata_csv,
        move_files=args.move
    )