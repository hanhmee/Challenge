#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def read_csv_rows(csv_path):
	with open(csv_path, "r", encoding="utf-8", newline="") as f:
		reader = csv.DictReader(f)
		rows = list(reader)
		return rows, reader.fieldnames or []


def combine_test_and_results(test_time_csv, results_csv, output_csv):
	test_rows, test_fields = read_csv_rows(test_time_csv)
	result_rows, result_fields = read_csv_rows(results_csv)

	required_test_fields = ["id", "canonical", "transcript"]
	required_result_fields = ["predict"]

	missing_test = [name for name in required_test_fields if name not in test_fields]
	missing_result = [name for name in required_result_fields if name not in result_fields]

	if missing_test:
		raise ValueError(f"Missing columns in test_time.csv: {missing_test}")
	if missing_result:
		raise ValueError(f"Missing columns in results.csv: {missing_result}")

	if len(test_rows) != len(result_rows):
		raise ValueError(
			"Row count mismatch: "
			f"test_time.csv has {len(test_rows)} rows, "
			f"results.csv has {len(result_rows)} rows."
		)

	output_fields = ["id", "predict", "canonical", "transcript"]

	with open(output_csv, "w", encoding="utf-8", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=output_fields)
		writer.writeheader()

		for test_row, result_row in zip(test_rows, result_rows):
			writer.writerow(
				{
					"id": test_row["id"],
					"predict": result_row["predict"],
					"canonical": test_row["canonical"],
					"transcript": test_row["transcript"],
				}
			)


def main():
	parser = argparse.ArgumentParser(
		description="Combine test_time.csv and results.csv into one CSV file."
	)
	parser.add_argument(
		"--test-time",
		default="test_time.csv",
		help="Path to test_time.csv (must have id, canonical, transcript)",
	)
	parser.add_argument(
		"--results",
		default="results.csv",
		help="Path to results.csv (must have predict)",
	)
	parser.add_argument(
		"--output",
		default="combined_test_results.csv",
		help="Path to output CSV",
	)
	args = parser.parse_args()

	combine_test_and_results(args.test_time, args.results, args.output)
	print(f"Wrote combined CSV to: {Path(args.output).resolve()}")


if __name__ == "__main__":
	main()
