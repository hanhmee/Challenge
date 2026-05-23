# MDD simple training framework

A structured repo for training MDD systems

## Structure

- `main.py`: CLI entry point, this is where you run the training process
- `trainer.py`: training loop, evaluation, inference, checkpointing
- `dataset.py`: dataset + collate functions
- `model.py`: model definitions
- `utils.py`: vocab, decoder, feature extractor, tensor conversion helpers
- `requirements.txt`: dependencies

## Usage

Create and setup a new conda environment:
```bash
conda create -n mdd python=3.8
conda activate mdd
pip install -r requirements.txt
```

Prepare your data to be in the same format with `train_time.csv`, `dev_time.csv`, `test_time.csv` (all 3 of them are in the same format), and your vocabulary into `vocab.json`

Modify the architecture inside `model.py`

Run training:

```bash
python main.py \
  --train_csv /path/to/train_time.csv \
  --dev_csv /path/to/dev_time.csv \
  --wav_dir /path/to/WAV \
  --vocab_path /path/to/vocab.json \
  --checkpoint_dir ./checkpoint
```