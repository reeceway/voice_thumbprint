#!/bin/bash
# Download ASVspoof 2019 LA dataset
# Warning: size is ~5GB

DATA_DIR="./data"
mkdir -p $DATA_DIR

echo "Downloading ASVspoof 2019 LA dataset..."
# Uses the path from asvspoof.org (hosted on datashare.ed.ac.uk)
# Note: verified URL for 2019 LA
URL="https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip"

if [ ! -f "$DATA_DIR/LA.zip" ]; then
    curl -L -o "$DATA_DIR/LA.zip" $URL
else
    echo "LA.zip already exists. Skipping download."
fi

echo "Extracting..."
unzip -q "$DATA_DIR/LA.zip" -d "$DATA_DIR"

echo "Done! Data is in $DATA_DIR/LA"
