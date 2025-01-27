
# Install additional Python packages
cd ~/scratch/macgyversemanticprobing
export HF_HOME=~/scratch/macgyversemanticprobing/.cache/huggingface
export HF_HUB_CACHE=~/scratch/macgyversemanticprobing/.cache/huggingface/hub
echo $HF_HOME
echo $HF_HUB_CACHE

pip install --upgrade pip
pip install numpy tenacity pandas matplotlib seaborn huggingface-hub transformers openai datasets torch accelerate>=0.26.0 chromadb sentence_transformers
pip install openpyxl scikit-learn 
pip install python-dotenv sentencepiece protobuf google bitsandbytes
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd MacGyverSemanticProbing
# Install dependencies from requirements file
pip install -r requirements.txt


# Display CUDA information
echo $CUDA_HOME
nvidia-smi && nvcc --version

echo "Done installing dependencies"

cd ~/macgyversemanticprobing/MacGyverSemanticProbing
