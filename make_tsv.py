from pathlib import Path
import csv

BASE_DIR = Path("/FilterFiles/Eichenbaum")
OUT_TSV = Path("output.tsv")

rows = []

for txt_path in BASE_DIR.rglob("mode_stdev_good_pairs.txt"):
    # Find the folder name right after "Eichenbaum"
    parts = txt_path.parts
    try:
        i = parts.index("Eichenbaum")
        folder_name = parts[i + 1]  # e.g. AJF016_CDEF1
    except (ValueError, IndexError):
        continue

    # Column 2: replace "_" with "/"
    col2 = folder_name.replace("_", "/")

    # Column 3: ../chensun/identify_stdp/data/Eichenbaum/AJF016/CDEF1/AJF016_CDEF1.pkl
    if "_" not in folder_name:
        continue
    part1, part2 = folder_name.split("_", 1)
    col3 = f"../chensun/identify_stdp/data/Eichenbaum/{part1}/{part2}/{folder_name}.pkl"

    # Column 4: ../BenR/prescreening/data/Eichenbaum/analysis/AJF016_CDEF1
    col4 = f"../BenR/prescreening/data/Eichenbaum/analysis/{folder_name}"

    with txt_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Input lines look like: T18a<TAB>T22a
            try:
                a, b = line.split()
            except ValueError:
                continue

            # Column 1: T18a:T22a
            col1 = f"{a}:{b}"

            # print it out
            print(col1)

            rows.append([col1, col2, col3, col4])

# Write TSV
with OUT_TSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_TSV}")
