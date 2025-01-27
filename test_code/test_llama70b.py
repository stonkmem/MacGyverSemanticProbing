from src.llama_funcs import *

def test_llama70b():
    prompt = "What is the capital of France?"
    responses, tokenlist, problist, hs = gen_prob()
