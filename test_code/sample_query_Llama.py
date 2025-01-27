from transformers import pipeline
from loginscript import hf_login    #create a script called loginscript.py and set it up as follows:
from llama_cpp import Llama
import numpy as np

'''
from huggingface_hub import login
def hf_login():
    login(token='')   <--CHANGE THIS!!!
'''

llm = Llama.from_pretrained(
    repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
    filename = 'Llama-3.2-1B-Instruct-Q6_K_L.gguf',
    logits_all = True,
    n_gpu_layers=-1
)

llm.create_chat_completion()