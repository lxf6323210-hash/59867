
import csv
import os
from pathlib import Path

from llama_verifier import LlamaVerifier


INPUT_CSV = Path("results/experiment_results_200.csv")
OUTPUT_CSV = Path("results/experiment_results_200_with_llama.csv")

NEW_COLUMNS = [
    "Llama Raw Verdict",
    "Llama Verdict",
    "Llama Evidence",
    "Llama Consistency",
    "Llama Raw Response",
    "Llama Status"
]

REQUIRED_COLUMNS = [
    "Image",
    "Caption Type",
    "Caption",
    "Similarity",
    "Qwen Verdict",
    "Qwen Evidence"
]


def get_record_key(row):
    image = str(row.get("Image", "")).replace("\\", "/").lower()
    caption_type = str(row.get("Caption Type", "")).strip().upper()
    caption = str(row.get("Caption", "")).strip()

    return image, caption_type, caption


def resolve_image_path(image_value):
    image_value = str(image_value).strip()
    image_value = image_value.replace("\\", os.sep)
    image_value = image_value.replace("/", os.sep)

    path = Path(image_value)

    if not path.is_absolute():
        path = Path.cwd() / path

    return path


def load_source_data():
    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Input CSV not found: {INPUT_CSV}"
        )

    with INPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise ValueError("The input CSV has no header.")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in fieldnames
    ]

    if missing:
        raise ValueError(
            f"Missing required CSV columns: {missing}"
        )

    if not rows:
        raise ValueError("The input CSV contains no records.")

    return fieldnames, rows


def load_previous_results():
    previous_results = {}

    if not OUTPUT_CSV.is_file():
        return previous_results

    with OUTPUT_CSV.open(
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            return previous_results

        missing = [
            column
            for column in NEW_COLUMNS
            if column not in reader.fieldnames
        ]

        if missing:
            raise ValueError(
                "Existing output CSV has an incompatible format. "
                f"Missing columns: {missing}"
            )

        for row in reader:
            if row.get("Llama Status") == "SUCCESS":
                previous_results[get_record_key(row)] = row

    return previous_results


def main():
    print("=" * 60)
    print("LLAMA BATCH EXPERIMENT")
    print("=" * 60)

    source_fields, source_rows = load_source_data()
    previous_results = load_previous_results()

    output_fields = list(source_fields)

    for column in NEW_COLUMNS:
        if column not in output_fields:
            output_fields.append(column)

    print("Input records:", len(source_rows))
    print("Previously completed:", len(previous_results))

    missing_images = []

    for index, row in enumerate(source_rows, start=1):
        image_path = resolve_image_path(row["Image"])

        if not image_path.is_file():
            missing_images.append(
                (index, str(image_path))
            )

        if not str(row.get("Caption", "")).strip():
            raise ValueError(
                f"Record {index} has an empty Caption."
            )

    if missing_images:
        print("\nMissing image files:")

        for index, path in missing_images:
            print(f"Record {index}: {path}")

        raise FileNotFoundError(
            "Some image files are missing. "
            "No batch inference was started."
        )

    pending = sum(
        1
        for row in source_rows
        if get_record_key(row) not in previous_results
    )

    print("Records to process:", pending)
    print("Output:", OUTPUT_CSV)

    verifier = None
    completed = 0
    failed = 0
    reused = 0

    try:
        with OUTPUT_CSV.open(
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=output_fields
            )

            writer.writeheader()
            file.flush()

            for index, row in enumerate(source_rows, start=1):

                key = get_record_key(row)

                if key in previous_results:
                    saved = previous_results[key]

                    for column in NEW_COLUMNS:
                        row[column] = saved.get(column, "")

                    reused += 1
                    writer.writerow(row)
                    file.flush()

                    print(
                        f"[{index}/{len(source_rows)}] "
                        f"REUSED: {row['Image']} "
                        f"({row['Caption Type']})"
                    )

                    continue

                if verifier is None:
                    print("\nLoading Llama model...")
                    verifier = LlamaVerifier()

                image_path = resolve_image_path(row["Image"])
                caption = row["Caption"]

                print(
                    f"\n[{index}/{len(source_rows)}] "
                    f"Processing: {row['Image']} "
                    f"({row['Caption Type']})"
                )

                try:
                    result = verifier.verify(
                        str(image_path),
                        caption
                    )

                    row["Llama Raw Verdict"] = result.get(
                        "raw_verdict", "UNCERTAIN"
                    )

                    row["Llama Verdict"] = result.get(
                        "verdict", "UNCERTAIN"
                    )

                    row["Llama Evidence"] = result.get(
                        "evidence", ""
                    )

                    row["Llama Consistency"] = result.get(
                        "consistency", ""
                    )

                    row["Llama Raw Response"] = result.get(
                        "raw_response", ""
                    )

                    row["Llama Status"] = "SUCCESS"

                    completed += 1

                    print(
                        "Raw Verdict:",
                        row["Llama Raw Verdict"]
                    )

                    print(
                        "Final Verdict:",
                        row["Llama Verdict"]
                    )

                    print(
                        "Consistency:",
                        row["Llama Consistency"]
                    )

                except Exception as error:
                    row["Llama Raw Verdict"] = ""
                    row["Llama Verdict"] = "UNCERTAIN"
                    row["Llama Evidence"] = str(error)
                    row["Llama Consistency"] = "INFERENCE_ERROR"
                    row["Llama Raw Response"] = ""
                    row["Llama Status"] = "ERROR"

                    failed += 1

                    print("ERROR:", str(error))

                writer.writerow(row)
                file.flush()

                print(
                    f"Progress: {index}/{len(source_rows)}"
                )

    finally:
        if verifier is not None:
            verifier.unload()

    print("\n" + "=" * 60)
    print("LLAMA BATCH EXPERIMENT COMPLETED")
    print("=" * 60)
    print("Total records:", len(source_rows))
    print("Newly completed:", completed)
    print("Reused records:", reused)
    print("Failed records:", failed)
    print("Output file:", OUTPUT_CSV)

    if failed:
        print(
            "Some records failed. Run this script again "
            "to retry unsuccessful records."
        )
    else:
        print("All records processed successfully.")


if __name__ == "__main__":
    main()
