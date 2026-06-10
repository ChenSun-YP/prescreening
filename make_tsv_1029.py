import os
from pathlib import Path
import csv
import random

BASE_DIR = Path("FilterFiles/Jan2010-Nonstationarity_Learning/1029")

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
        i = parts.index("Jan2010-Nonstationarity_Learning")
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
    col3 = (
        f"data/Jan2010-Nonstationarity_Learning/Rodent_WFU_DNMS/1029/{neuron_name}.pkl"
    )

    print(f"Extracted neuron name: {neuron_name}")
    col4 = f"data/Jan2010-Nonstationarity_Learning/Rodent_WFU_DNMS/single_pair_analysis/1029/{neuron_name}"

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


if not SAMPLE_FROM_EACH:
    print(len(rows), "total rows found.")

    # --- Randomly select 20 pairs with a fixed seed (Im just using the date) ---
    # SEED = 20260401
    SEED = 20260404
    N = 100

    OUT_TSV = Path(f"Control_100_pair_{SEED}.tsv")

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
