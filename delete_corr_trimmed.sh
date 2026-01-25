#!/bin/bash
#SBATCH --job-name=copy_ranks
#SBATCH --output=copy_ranks.out
#SBATCH --error=copy_ranks.err
#SBATCH --time=00:10:00
#SBATCH --partition=main
#SBATCH --ntasks=1


# -------- USER SETTINGS --------
SOURCE_DIR="/home1/benren/prescreening"
# --------------------------------

# Move into the source directory
cd "$SOURCE_DIR"

echo "The following files will be deleted:"
rm -f corr_trimmed*
echo done