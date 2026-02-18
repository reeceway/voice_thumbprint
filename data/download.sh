#!/bin/bash
# Download ASVspoof 2019 LA dataset
# Warning: size is ~7GB compressed, ~14GB extracted

DATA_DIR="$(dirname "$0")"
mkdir -p $DATA_DIR

echo "Downloading ASVspoof 2019 LA dataset..."
echo "This is ~7GB and may take 15-30 minutes depending on connection."

# Uses the path from asvspoof.org (hosted on datashare.ed.ac.uk)
URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"

# Download with resume support (-C -) and progress
if [ -f "$DATA_DIR/LA.zip" ]; then
    echo "Resuming download..."
    curl -C - -L -o "$DATA_DIR/LA.zip" $URL
else
    echo "Starting fresh download..."
    curl -L -o "$DATA_DIR/LA.zip" $URL
fi

if [ $? -ne 0 ]; then
    echo "Download failed. You can resume by running this script again."
    exit 1
fi

echo "Download complete!"
echo "Extracting (this may take a few minutes)..."

# Extract with overwrite
unzip -o -q "$DATA_DIR/LA.zip" -d "$DATA_DIR"

echo "Done! Data is in $DATA_DIR/LA"
echo ""
echo "Structure:"
echo "  $DATA_DIR/LA/ASVspoof2019_LA_train/  - Training set"
echo "  $DATA_DIR/LA/ASVspoof2019_LA_dev/    - Development set"
echo "  $DATA_DIR/LA/ASVspoof2019_LA_eval/   - Evaluation set"
echo "  $DATA_DIR/LA/*.txt                   - Protocol files"
