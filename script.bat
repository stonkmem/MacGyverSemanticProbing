nvidia-smi && nvcc --version
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
set FORCE_CMAKE=1
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose
pip install numpy tenacity pandas matplotlib seaborn huggingface-hub tranformers openai