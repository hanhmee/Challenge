import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


@dataclass
class ManifestSpec:
    path: Path
    split: str
    dataset_name: str


def _ensure_string_tokens(value) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            parsed = ast.literal_eval(stripped)
            return [str(v) for v in parsed]
        return [t for t in stripped.split() if t]
    return []


def _build_error(canonical_tokens: List[str], transcript_tokens: List[str]) -> List[int]:
    n = max(len(canonical_tokens), len(transcript_tokens))
    errors = []
    for i in range(n):
        c = canonical_tokens[i] if i < len(canonical_tokens) else '<pad>'
        t = transcript_tokens[i] if i < len(transcript_tokens) else '<pad>'
        errors.append(1 if c == t else 0)
    return errors


def _build_canonical_time(tokens: List[str]) -> List[Dict[tuple, str]]:
    # Placeholder sequential alignment. Suitable for WL mode; for MFA mode provide true alignment.
    return [{(i, i + 1): token} for i, token in enumerate(tokens)]


def _normalize_row(row: pd.Series, dataset_name: str) -> Dict:
    audio_path = row.get('AudioPath', row.get('audio_path', row.get('audio', row.get('wav_path', ''))))
    path_value = row.get('Path', row.get('path', ''))

    if not path_value and audio_path:
        path_value = Path(str(audio_path)).with_suffix('').as_posix()

    canonical_src = row.get('Canonical', row.get('canonical_phonemes', row.get('canonical', row.get('phonemes', ''))))
    transcript_src = row.get('Transcript', row.get('phonemes', row.get('perceived_phonemes', row.get('transcript', ''))))

    canonical_tokens = _ensure_string_tokens(canonical_src)
    transcript_tokens = _ensure_string_tokens(transcript_src)

    if not canonical_tokens or not transcript_tokens:
        return {}

    canonical_time = row.get('Canonical_time', row.get('canonical_time', None))
    if canonical_time is None or canonical_time == '':
        canonical_time = _build_canonical_time(canonical_tokens)
    elif isinstance(canonical_time, str):
        canonical_time = ast.literal_eval(canonical_time)

    error = row.get('Error', row.get('error', None))
    if error is None or error == '':
        error = _build_error(canonical_tokens, transcript_tokens)
    elif isinstance(error, str):
        error = ast.literal_eval(error)

    normalized = {
        'Dataset': dataset_name,
        'Path': str(path_value),
        'AudioPath': str(audio_path) if audio_path else '',
        'Canonical': ' '.join(canonical_tokens),
        'Transcript': ' '.join(transcript_tokens),
        'Canonical_time': str(canonical_time),
        'Error': str(error),
    }
    return normalized


def load_and_normalize_manifest(spec: ManifestSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.path)
    records = []
    for _, row in df.iterrows():
        normalized = _normalize_row(row, dataset_name=spec.dataset_name)
        if normalized:
            normalized['Split'] = spec.split
            records.append(normalized)
    return pd.DataFrame(records)


def build_training_csvs(manifest_specs: List[ManifestSpec], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = [load_and_normalize_manifest(spec) for spec in manifest_specs]
    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    train_df = merged[merged['Split'] == 'train'].drop(columns=['Split'])
    dev_df = merged[merged['Split'].isin(['dev', 'val', 'valid'])].drop(columns=['Split'])
    test_df = merged[merged['Split'] == 'test'].drop(columns=['Split'])

    train_df.to_csv(output_dir / 'train_time.csv')
    dev_df.to_csv(output_dir / 'dev_time.csv')
    test_df.to_csv(output_dir / 'test_time.csv')

    summary = {
        'train': len(train_df),
        'dev': len(dev_df),
        'test': len(test_df),
        'datasets': sorted(list(merged['Dataset'].unique())) if not merged.empty else [],
    }

    return summary


def parse_manifest_args(manifest_args: List[str]) -> List[ManifestSpec]:
    specs: List[ManifestSpec] = []
    for item in manifest_args:
        # format: split|dataset|path
        parts = item.split('|', 2)
        if len(parts) != 3:
            raise ValueError(f"Invalid --manifest format: {item}. Expected split|dataset|path")
        split, dataset_name, path = parts
        specs.append(ManifestSpec(path=Path(path), split=split.lower(), dataset_name=dataset_name))
    return specs


def build_args():
    parser = argparse.ArgumentParser(description='Prepare unified train/dev/test csv from notebook-style manifests')
    parser.add_argument(
        '--manifest',
        action='append',
        required=True,
        help='Manifest spec: split|dataset|path (example: train|L2-ARCTIC|data/l2_train.csv)',
    )
    parser.add_argument('--output_dir', type=str, default='./prepared_data')
    return parser.parse_args()


def main():
    args = build_args()
    specs = parse_manifest_args(args.manifest)
    summary = build_training_csvs(specs, output_dir=Path(args.output_dir))
    print('Prepared data summary:')
    print(summary)


if __name__ == '__main__':
    main()
