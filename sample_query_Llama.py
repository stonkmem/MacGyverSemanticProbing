from transformers import pipeline
from loginscript import hf_login

hf_login()


messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")
pipe(messages)