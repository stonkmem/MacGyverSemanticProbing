nvidia-smi && nvcc --version
set CMAKE_ARGS=-DLLAMA_CUBLAS=on
set FORCE_CMAKE=1
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose
pip install numpy tenacity pandas matplotlib seaborn huggingface-hub tranformers openai
pip install llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
pip install --no-cache-dir llama-cpp-python==0.2.90 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123
pip install llama-cpp-python huggingface-hub transformers
nvcc --version
echo $CUDA_HOME
CMAKE_ARGS="-DLLAMA_CUDA=on -DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir


%%capture
pip install unsloth
@REM # Also get the latest nightly Unsloth!
pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps "xformers<0.0.27" peft accelerate bitsandbytes
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124
pip install Transformers