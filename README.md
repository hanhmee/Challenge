# xlsr_mdd_structured

A structured rewrite of the XLSR MDD training workflow from `xlsr-accent-training.ipynb`, using
`Refining-Linguistic-Information-Utilization-MDD` as the code base.

This repo keeps the training and processing logic equivalent to the original scripts:
- `train_wl.py` (WL mode)
- `train_MFA.py` (MFA mode)
- `MDD_model.py`
- `dataloader.py`

## Structure

- `main.py`: CLI entry point
- `trainer.py`: training loop, evaluation, checkpointing
- `dataset.py`: dataset + collate functions for WL/MFA
- `model.py`: model definitions
- `utils.py`: vocab, decoder, feature extractor, tensor conversion helpers
- `data_prep.py`: multi-dataset manifest fusion to `train_time.csv`/`dev_time.csv`/`test_time.csv`
- `xlsr_mdd_pipeline.ipynb`: clean orchestration notebook for data prep + training
- `requirements.txt`: dependencies

## Flow Parity Notes

- Same as Refining repo:
  - WL/MFA model definitions and CTC training/evaluation loop
  - Input CSV schema expected by the training code
- Not identical to the original long notebook by default:
  - The original notebook includes raw dataset parsing (TIMIT/LSVSC/L2-ARCTIC), Vietnamese G2P, and HuggingFace trainer utilities.
  - This structured repo focuses on reproducible training once manifests are prepared.

To reproduce the notebook-style multi-dataset training flow, first build manifests from all datasets, then run `data_prep.py` to normalize and merge them.

## Multi-Dataset Preparation

Prepare one or more manifest CSV files per split (`train`, `dev` or `val`, `test`) and run:

```bash
python data_prep.py \
  --manifest "train|L2-ARCTIC|/path/to/l2_train_manifest.csv" \
  --manifest "dev|L2-ARCTIC|/path/to/l2_dev_manifest.csv" \
  --manifest "test|L2-ARCTIC|/path/to/l2_test_manifest.csv" \
  --manifest "train|LSVSC|/path/to/lsvsc_train_manifest.csv" \
  --manifest "dev|LSVSC|/path/to/lsvsc_valid_manifest.csv" \
  --manifest "test|LSVSC|/path/to/lsvsc_test_manifest.csv" \
  --output_dir ./prepared_data
```

Then train with:

```bash
python main.py --mode wl \
  --train_csv ./prepared_data/train_time.csv \
  --dev_csv ./prepared_data/dev_time.csv \
  --wav_dir "" \
  --vocab_path ./vocab.json \
  --checkpoint_dir ./checkpoint
```

Notes:
- `dataset.py` accepts both `Path` (legacy) and `AudioPath`/`audio_path` columns.
- Use `wl` mode when you do not have true alignment spans in `Canonical_time`.
- Use `mfa` mode only when `Canonical_time` is accurate frame alignment.

## Usage

Install dependencies:

```bash
pip install -r requirements.txt
```

Train with MFA mode (equivalent to old `train_MFA.py` flow):

```bash
python main.py --mode mfa \
  --train_csv /path/to/train_time.csv \
  --dev_csv /path/to/dev_time.csv \
  --wav_dir /path/to/WAV \
  --vocab_path /path/to/vocab.json \
  --checkpoint_dir ./checkpoint
```

Train with WL mode (equivalent to old `train_wl.py` flow):

```bash
python main.py --mode wl \
  --train_csv /path/to/train_time.csv \
  --dev_csv /path/to/dev_time.csv \
  --wav_dir /path/to/WAV \
  --vocab_path /path/to/vocab.json \
  --checkpoint_dir ./checkpoint
```

## Notes

- Paths are intentionally configurable because dataset locations can change.
- `blank`/`pad` token behavior and CTC decoding setup follow the original implementation.
