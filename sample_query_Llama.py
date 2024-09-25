from transformers import pipeline
from loginscript import hf_login    #create a script called loginscript.py and set it up as follows:

'''
from huggingface_hub import login
def hf_login():
    login(token='')   <--CHANGE THIS!!!
'''


hf_login()


messages = [
    {"role": "user", "content": "Who are you?"},
]
pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3.1-8B-Instruct")
pipe(messages)