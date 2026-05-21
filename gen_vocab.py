import json

input_file = "/home4/datpt/trungnt/mdd/VMD-VLSP2023/VMD-VLSP23-training-set/metadata/lexicon_vmd.txt"
output_file = "vocab.json"

phonemes = set()

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) > 1:
            phonemes.update(parts[1:])

vocab = {phn: idx for idx, phn in enumerate(sorted(phonemes))}

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print(f"Saved {len(vocab)} phonemes to {output_file}")