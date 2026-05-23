import os
from typing import Dict

import librosa
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils import (
    PAD_TOKEN_ID,
    SAMPLE_RATE,
    text_to_tensor,
)


class MDDDataset(Dataset):
    def __init__(self, data: pd.DataFrame, wav_dir: str, vocab: Dict[str, int]):
        self.data = data
        self.len_data = len(data)
        self.path = self._get_column(data, ['Path', 'path'])
        self.audio_path = self._get_column(data, ['AudioPath', 'audio_path'])
        self.canonical = self._get_column(data, ['Canonical', 'canonical'])
        self.transcript = self._get_column(data, ['Transcript', 'transcript'])
        self.wav_dir = wav_dir
        self.vocab = vocab

    def __getitem__(self, index):
        wav_path = self._resolve_wav_path(index)
        waveform, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
        linguistic = text_to_tensor(self.canonical[index], self.vocab)
        transcript = text_to_tensor(self.transcript[index], self.vocab)
        return waveform, linguistic, transcript

    def __len__(self):
        return self.len_data

    @staticmethod
    def _get_column(data: pd.DataFrame, names, default=None):
        for name in names:
            if name in data.columns:
                return list(data[name])
        return [default] * len(data)

    def _resolve_wav_path(self, index: int) -> str:
        if self.audio_path[index]:
            audio_path = str(self.audio_path[index])
            return audio_path

        raw_path = str(self.path[index])
        if raw_path.lower().endswith('.wav'):
            return os.path.join(self.wav_dir, raw_path)
        return os.path.join(self.wav_dir, f"{raw_path}.wav")


def make_collate_fn(feature_extractor, device: torch.device):
    def collate_fn(batch):
        with torch.no_grad():
            max_col = [-1] * 4

            for row in batch:
                max_col[0] = max(max_col[0], row[0].shape[0])
                max_col[1] = max(max_col[1], len(row[1]))
                max_col[2] = max(max_col[2], len(row[2]))

            cols = {
                'waveform': [],
                'linguistic': [],
                'transcript': [],
                'outputlengths': [], 
                'input_lengths': [],
            }

            for row in batch:
                cols['input_lengths'].append(row[0].shape[0]) 

                pad_wav = np.concatenate([row[0], np.zeros(max_col[0] - row[0].shape[0])])
                cols['waveform'].append(pad_wav)

                row[1].extend([PAD_TOKEN_ID] * (max_col[1] - len(row[1])))
                cols['linguistic'].append(row[1])

                cols['outputlengths'].append(len(row[2]))
                row[2].extend([PAD_TOKEN_ID] * (max_col[2] - len(row[2])))
                cols['transcript'].append(row[2])

            inputs = feature_extractor(cols['waveform'], sampling_rate=SAMPLE_RATE)
            input_values = torch.tensor(inputs.input_values, device=device)

            cols['waveform'] = torch.tensor(cols['waveform'], dtype=torch.float, device=device)
            cols['linguistic'] = torch.tensor(cols['linguistic'], dtype=torch.long, device=device)
            cols['transcript'] = torch.tensor(cols['transcript'], dtype=torch.long, device=device)
            cols['outputlengths'] = torch.tensor(cols['outputlengths'], dtype=torch.long, device=device)
            cols['input_lengths'] = torch.tensor(cols['input_lengths'], dtype=torch.long, device=device)

            return (
                input_values,
                cols['linguistic'],
                cols['transcript'],
                cols['outputlengths'],
                cols['input_lengths'],  
            )

    return collate_fn