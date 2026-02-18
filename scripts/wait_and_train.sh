#!/bin/bash
# Monitor download and automatically start training when complete

DATA_DIR="/Users/reeceway/Desktop/voiceprint/voice_thumbprint/data"
ZIP_FILE="$DATA_DIR/LA.zip"
EXPECTED_SIZE_MB=7000  # ~7GB

echo "Monitoring ASVspoof download..."
echo "Will start training when download completes."
echo ""

while true; do
    if [ -f "$ZIP_FILE" ]; then
        SIZE=$(stat -f%z "$ZIP_FILE" 2>/dev/null || stat -c%s "$ZIP_FILE" 2>/dev/null)
        SIZE_MB=$((SIZE / 1024 / 1024))
        
        echo -ne "\rDownloaded: ${SIZE_MB}MB / ~${EXPECTED_SIZE_MB}MB ($((SIZE_MB * 100 / EXPECTED_SIZE_MB))%)"
        
        # Check if download seems complete (no curl process running and size is reasonable)
        if ! pgrep -f "curl.*LA.zip" > /dev/null && [ $SIZE_MB -gt 6500 ]; then
            echo ""
            echo ""
            echo "✓ Download appears complete!"
            break
        fi
    fi
    sleep 30
done

# Extract
echo ""
echo "Extracting dataset..."
cd "$DATA_DIR"
unzip -o -q LA.zip

if [ $? -eq 0 ]; then
    echo "✓ Extraction complete!"
    echo ""
    
    # Start training
    cd /Users/reeceway/Desktop/voiceprint/voice_thumbprint
    
    echo "Starting model training..."
    echo ""
    
    # Train all models
    /usr/bin/python3 scripts/train_all_models.py \
        --data-dir ./data/LA \
        --gmm-components 256 \
        --classifier xgboost \
        --output-dir models
    
    echo ""
    echo "Training complete! Starting evaluation..."
    
    # Evaluate
    /usr/bin/python3 scripts/evaluate.py \
        --data-dir ./data/LA \
        --models-dir models \
        --output evaluation_results.json
    
    echo ""
    echo "✓ All done! Check evaluation_results.json for results."
    
else
    echo "❌ Extraction failed!"
    exit 1
fi
