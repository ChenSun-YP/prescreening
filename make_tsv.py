from pathlib import Path
import csv

BASE_DIR = Path("/FilterFiles/Eichenbaum")
OUT_TSV = Path("output.tsv")

rows = []

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

# Write TSV
with OUT_TSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUT_TSV}")
