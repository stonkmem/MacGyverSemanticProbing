#!/bin/bash

# Clone the repository
# git clone https://github.com/stonkmem/MacGyverSemanticProbing.git

# # Change directory to the repository
# cd MacGyverSemanticProbing

# scp .env <aspire_user>@jumphost.ntu.edu.sg:/path/on/jumphost/.env /home/<your_username>/macgyversemanticprobing/.env

# Set environment variables
# export CMAKE_ARGS="-DGGML-CUDA=ON"
# export FORCE_CMAKE=1

# Install llama-cpp-python
# pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose

# Install additional Python packages
cd ~/scratch
export HF_HOME=~/scratch/macgyversemanticprobing/.cache/huggingface
export HF_HUB_CACHE=~/scratch/macgyversemanticprobing/.cache/huggingface/hub
echo $HF_HOME
echo $HF_HUB_CACHE

pip install --upgrade pip
pip install numpy tenacity pandas matplotlib seaborn huggingface-hub transformers openai datasets torch accelerate>=0.26.0 
pip install openpyxl scikit-learn 
pip install python-dotenv sentencepiece protobuf google bitsandbytes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install dependencies from requirements file
pip install -r requirements.txt


# Display CUDA information
echo $CUDA_HOME
nvidia-smi && nvcc --version

echo "Done installing dependencies"

cd ~/macgyversemanticprobing/MacGyverSemanticProbing
