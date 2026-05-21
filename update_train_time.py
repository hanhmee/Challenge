#!/usr/bin/env python3
"""Update train_time.csv: prefix `path` entries and drop columns.

Usage:
  python update_train_time.py --file train_time.csv [--backup]
"""
import argparse
import csv
import os
import shutil
from datetime import datetime


PREFIX = "en-mdd/EN_MDD/WAV"


def process_csv(path, backup=True):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    dirpath = os.path.dirname(path) or "."
    basename = os.path.basename(path)

    if backup:
        ts = datetime.now().strftime("%Y%m%d%H%M%S")
        bak_name = f"{basename}.bak.{ts}"
        bak_path = os.path.join(dirpath, bak_name)
        shutil.copy2(path, bak_path)
        print(f"Backup created: {bak_path}")

    tmp_path = os.path.join(dirpath, f"{basename}.tmp")

    with open(path, newline='', encoding='utf-8') as rf:
        reader = csv.DictReader(rf)
        # Determine output fieldnames: remove Canonical_time and Error if present
        fieldnames = [f for f in reader.fieldnames if f not in ("Canonical_time", "Error")]
        if "path" not in fieldnames:
            raise KeyError("CSV does not contain a 'path' column")

        with open(tmp_path, 'w', newline='', encoding='utf-8') as wf:
            writer = csv.DictWriter(wf, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                # Prefix path if not already prefixed
                orig = (row.get('path') or '').strip()
                # remove leading slashes to keep join consistent
                orig_stripped = orig.lstrip('/\\')
                if orig_stripped and not orig_stripped.startswith(PREFIX):
                    new_path = f"{PREFIX}/{orig_stripped}"
                else:
                    new_path = orig

                row['path'] = new_path
                # Remove unwanted keys if present
                if 'Canonical_time' in row:
                    row.pop('Canonical_time', None)
                if 'Error' in row:
                    row.pop('Error', None)

                # Write only the selected fieldnames
                out_row = {k: row.get(k, '') for k in fieldnames}
                writer.writerow(out_row)

    # Replace original
    shutil.move(tmp_path, path)
    print(f"Updated file written to: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', '-f', default='train_time.csv', help='Path to train_time.csv')
    parser.add_argument('--no-backup', dest='backup', action='store_false', help='Do not create a backup')
    args = parser.parse_args()

    process_csv(args.file, backup=args.backup)


if __name__ == '__main__':
    main()
