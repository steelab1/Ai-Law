#!/bin/bash

# Step 0: Create a Python virtual environment
echo "Creating Python virtual environment..."
python3 -m venv .venv || { echo "Failed to create virtual environment"; exit 1; }

# Step 1: Activate the virtual environment
echo "Activating the virtual environment..."
source .venv/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }

# Step 2: Install PyTorch FIRST (required by FlagEmbedding)
echo "Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    pip install torch || { echo "Failed to install PyTorch"; exit 1; }
else
    pip install torch --index-url https://download.pytorch.org/whl/cu121 || { echo "Failed to install PyTorch"; exit 1; }
fi

# Step 3: Change directory to FlagEmbedding
echo "Changing directory to FlagEmbedding..."
cd FlagEmbedding || { echo "Directory FlagEmbedding not found"; exit 1; }

# Step 4: Install the package (without finetune extras on Mac - flash-attn not supported)
echo "Installing FlagEmbedding package..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # Mac: Install basic package without flash-attn dependency
    pip install -e . || { echo "Failed to install the package"; exit 1; }
else
    pip install -e .[finetune] || { echo "Failed to install the package"; exit 1; }
fi

echo "Environment setup completed successfully!"
