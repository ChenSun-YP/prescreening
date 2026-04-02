import os
from pathlib import Path
import csv
import random

BASE_DIR = Path("FilterFiles/Eichenbaum")
OUT_TSV = Path("output.tsv")

rows = []

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
        i = parts.index("Eichenbaum")
        folder_name = parts[i + 1]
        print(f"Extracted folder name: {folder_name}")
    except (ValueError, IndexError):
        continue

    # Column 2
    col2 = folder_name.replace("_", "/")
    print(f"Constructed col2: {col2}")

    # Column 3 + 4
    if "_" not in folder_name:
        continue
    part1, part2 = folder_name.split("_", 1)

    col3 = f"../chensun/identify_stdp/data/Eichenbaum/{part1}/{part2}/{folder_name}.pkl"
    col4 = f"../BenR/prescreening/data/Eichenbaum/analysis/{folder_name}"

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


# --- Randomly select 20 pairs with a fixed seed ---
SEED = 20260401
N = 20

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
