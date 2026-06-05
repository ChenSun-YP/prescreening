import os
from pathlib import Path
import csv
import random

BASE_DIR = Path(
    "FilterFiles/sz_dataset_kilosorted/2025_05_15hippocampus"
)  # CHANGE HERE: keep the FilterFiles directory, but change the rest to match the "file_dir" & "pkl_file" in config_dnms.json but remove the .pkl at the end
# if this errors out, look for the FilterFiles directory & change the path accordingly

rows = []

SAMPLE_FROM_EACH = True

print(f"Searching for files in {BASE_DIR}...")

files = os.listdir(BASE_DIR)

for file in files:
    print(file)

print(
    f"BASE_DIR.rglob('mode_stdev_good_pairs.txt'): {list(BASE_DIR.rglob('mode_stdev_good_pairs.txt'))}"
)

for txt_path in BASE_DIR.rglob("mode_stdev_good_pairs.txt"):
    parts = txt_path.parts
    print(f"Processing {txt_path} with parts: {parts}")

    # Only keep files inside a "semiFine" directory
    if "semiFine" not in parts:
        continue

    # Extract AJF016_CDEF1 (folder right after "Eichenbaum")
    try:
        i = parts.index(
            "sz_dataset_kilosorted"
        )  # CHANGE HERE: should be same as "file_dir" from the config_dnms.json
        folder_name = parts[i + 1]
        print(f"Extracted folder name: {folder_name}")
    except (ValueError, IndexError):
        continue

    # Column 2
    col2 = folder_name.replace("_", "/")
    print(f"Constructed col2: {col2}")

    # Column 3 + 4

    neuron_name = parts[4]
    print(f"Extracted neuron name: {neuron_name}")
    col3 = f"data/sz_dataset_kilosorted/2025_05_15hippocampus/{neuron_name}.pkl"

    print(f"Extracted neuron name: {neuron_name}")
    col4 = f"data/sz_dataset_kilosorted/single_pair_analysis/2025_05_15hippocampus/{neuron_name}"

    # Read file
    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                a, b = line.split()
            except ValueError:
                continue

            col1 = f"{a}:{b}"

            # print each pair
            print(col1)

            rows.append([col1, col2, col3, col4])

        if SAMPLE_FROM_EACH:

            print(len(rows), "total rows found.")

            # --- Randomly select 20 pairs with a fixed seed (Im just using the date) ---
            # SEED = 20260401
            SEED = 20260404
            N = 20

            OUT_TSV = Path(f"kilosort_20_pair_{SEED}.tsv")

            random.seed(SEED)

            if len(rows) > N:
                rows = random.sample(rows, N)  # pick 20 unique rows
            else:
                print(f"Only {len(rows)} rows available, using all.")

            # Write TSV
            with OUT_TSV.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter="\t")
                writer.writerows(rows)

            print(f"Wrote {len(rows)} rows to {OUT_TSV}")


if not SAMPLE_FROM_EACH:
    print(len(rows), "total rows found.")

    # --- Randomly select 20 pairs with a fixed seed (Im just using the date) ---
    # SEED = 20260401
    SEED = 20260404
    N = 20

    OUT_TSV = Path(f"Control_20_pair_{SEED}.tsv")

    random.seed(SEED)

    if len(rows) > N:
        rows = random.sample(rows, N)  # pick 20 unique rows
    else:
        print(f"Only {len(rows)} rows available, using all.")

    # Write TSV
    with OUT_TSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {OUT_TSV}")
