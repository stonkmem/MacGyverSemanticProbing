nvidia-smi && nvcc --version
set CMAKE_ARGS="-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS -DGGML_CUBLABS=ON -DGGML-CUDA=ON"
set FORCE_CMAKE=1
pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose
pip install numpy tenacity pandas matplotlib seaborn huggingface-hub tranformers openai
@REM pip install llama-cpp-python
@REM CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
@REM pip install --no-cache-dir llama-cpp-python==0.2.90 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu123
@REM pip install llama-cpp-python huggingface-hub transformers
@REM nvcc --version
echo $CUDA_HOME
pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir


@REM %%capture
@REM pip install unsloth
@REM @REM # Also get the latest nightly Unsloth!
@REM pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
@REM pip install --no-deps "xformers<0.0.27" peft accelerate bitsandbytes
@REM pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu124
@REM pip install Transformers