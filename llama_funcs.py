from llama_cpp import Llama
import sys
import os
import transformers
import torch
from helper_funcs import *
from data import *
from helper_funcs import gen_chat_object

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
load_dotenv()

huggingface_token = os.getenv("HF_TOKEN")



if __name__ == '__main__':
    if sys.argv[1] == 'llama':
        modelpath = "meta-llama/Llama-3.1-8B-Instruct"
        # llm = Llama.from_pretrained(
        #     repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        #     filename = 'Llama-3.2-3B-Instruct-Q6_K_L.gguf',
        #     logits_all = True,
        #     n_gpu_layers=-1
        # )
        # wipe_llm = llm.save_state()

        # entailment_llm = Llama.from_pretrained(
        #     repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        #     filename = 'Llama-3.2-3B-Instruct-Q6_K_L.gguf',
        #     logits_all = True,
        #     n_gpu_layers=-1
        # )
        # wipe_entailment_llm = entailment_llm.save_state()

        # llm_fact = Llama.from_pretrained(
        #     repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        #     filename = 'Llama-3.2-3B-Instruct-Q6_K_L.gguf',
        #     logits_all = True,
        #     n_gpu_layers=-1
        # )
        # wipe_llm_fact = llm_fact.save_state()

    elif sys.argv[1] == 'vicuna':
        modelpath = "lmsys/vicuna-7b-v1.5"
        # llm = Llama.from_pretrained(
        #     repo_id="TheBloke/stable-vicuna-13B-GGUF",
        #     filename = 'stable-vicuna-13B.Q6_K.gguf',
        #     logits_all = True,
        #     n_gpu_layers=-1
        # )
        # wipe_llm = llm.save_state()

        # entailment_llm = Llama.from_pretrained(
        #     repo_id="TheBloke/stable-vicuna-13B-GGUF",
        #     filename = 'stable-vicuna-13B.Q6_K.gguf',
        #     logits_all = True,
        #     n_gpu_layers=-1
        # )
        # wipe_entailment_llm = entailment_llm.save_state()

        # llm_fact = Llama.from_pretrained(
        #     repo_id="TheBloke/stable-vicuna-13B-GGUF",
        #     filename = 'stable-vicuna-13B.Q6_K.gguf',
        #     logits_all = True,
        #     n_gpu_layers=-1
        # )
        # wipe_llm_fact = llm_fact.save_state()
    elif sys.argv[1] == 'mistral':
        modelpath = "mistralai/Mixtral-8x22B-Instruct-v0.1"
        # llm = Llama.from_pretrained(
        #     repo_id="bartowski/Mistral-22B-v0.2-GGUF",
        #     filename="Mistral-22B-v0.2-Q5_K_M.gguf",
        #     logits_all = True,
        #     n_gpu_layers = -1
        # )
        # wipe_llm = llm.save_state()

        # entailment_llm = Llama.from_pretrained(
        #     repo_id="bartowski/Mistral-22B-v0.2-GGUF",
        #     filename="Mistral-22B-v0.2-Q5_K_M.gguf",
        #     logits_all = True,
        #     n_gpu_layers = -1
        # )
        # wipe_entailment_llm = entailment_llm.save_state()

        # llm_fact = Llama.from_pretrained(
        #     repo_id="bartowski/Mistral-22B-v0.2-GGUF",
        #     filename="Mistral-22B-v0.2-Q5_K_M.gguf",
        #     logits_all = True,
        #     n_gpu_layers = -1
        # )
        # wipe_llm_fact = llm_fact.save_state()
    else:
        modelpath = "meta-llama/Llama-3.1-8B-Instruct"
        # llm = Llama.from_pretrained(
        #     repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        #     filename = 'Llama-3.2-3B-Instruct-Q6_K_L.gguf',
        #     logits_all = True,
        #     n_gpu_layers=-1
        # )
        # wipe_llm = llm.save_state()

        # entailment_llm = Llama.from_pretrained(
        #     repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        #     filename = 'Llama-3.2-3B-Instruct-Q6_K_L.gguf',
        #     logits_all = True,
        #     n_gpu_layers=-1
        # )
        # wipe_entailment_llm = entailment_llm.save_state()

        # llm_fact = Llama.from_pretrained(
        #     repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        #     filename = 'Llama-3.2-3B-Instruct-Q6_K_L.gguf',
        #     logits_all = True,
        #     n_gpu_layers=-1
        # )
        # wipe_llm_fact = llm_fact.save_state()
    tokenizer = AutoTokenizer.from_pretrained(modelpath) 
 
    model = AutoModelForCausalLM.from_pretrained(modelpath)
    model.to("cuda")


    # hfclient = InferenceClient(api_key=huggingface_token)
    

# def get_entailment_llama(question, a, b):
#     entailment_llm.reset()
#     return entailment_llm.create_chat_completion(
#         messages=[
#             {'role': 'system', 'content': 'You are a skilled linguist studying semantic entailment. As a task, you will have to determine whether one sentence semantically entails another.'},
#             {'role': 'user', 'content': f'We are evaluating answers to the question: {question}.\n Here are 2 possible answers:\nSentence A: {a}\nSentence B: {b}\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, entailment, or neutral.'}
#         ]
#     )

# def get_corr_feas_eff_llama(fraege, antworten):
#     llm_fact.reset()
#     ratings = []
#     for i in range(len(fraege)):
#         ratings.append(llm_fact.create_chat_completion(messages=[
#                 {'role':'system', 'content':'Please act as an impartial judge and evaluate the quality\
# of the solution provided by an AI assistant to the user problem displayed below. \
# Your evaluation should rate the feasability and efficiency of the response. Be as objective as possible. \n\
# After providing your explanation, please state whether the response is or is not effective or feasible by strictly following this format: "Feasibility: [[YES/NO]], Efficiency: [[YES/NO]]", for example: "Feasibility: [[YES]], Efficiency: [[NO]]".'},
#                 {'role':'user','content':f"""[Question]
# {fraege[i]}
    
# [The Start of Assistant’s Answer]
# {antworten[i]}
# [The End of Assistant's Answer]"""}
#             ])['choices'][0]['message']['content'])
#     return ratings

# def get_P(x, y):
#     llm.reset()
#     token_x = llm.tokenize(x.encode('utf-8'), special=True)
#     token_y = llm.tokenize(y.encode('utf-8'), special=True, add_bos=False)

#     logprobs=[]
#     logits=[]
#     curr = token_x[:]

#     llm.eval(curr)
#     for token in token_y:
#         curr.append(token)
#         logprobs.append(llm.logits_to_logprobs(llm.eval_logits)[-1][token])
#         logits.append(llm.eval_logits[-1][token])
#         llm.eval([token])

#     return x, token_x, y, token_y, logprobs, logits

def convert_openai_to_llama_prompt(ls):
    pmpt = '<|begin_of_text|>'
    for msg in ls:
        if msg['role'] == 'user':
            pmpt += '<|start_header_id|>user<|end_header_id|>\n\n'
            pmpt += msg['content'] + '<|eot_id|>'
        elif msg['role'] == 'assistant':
            pmpt += '<|start_header_id|>assistant<|end_header_id|>\n\n'
            pmpt += msg['content'] + '<|eot_id|>'
        elif msg['role'] == 'system':
            pmpt += '<|start_header_id|>system<|end_header_id|>\n\n'
            pmpt += msg['content'] + '<|eot_id|>'
    pmpt += '<|start_header_id|>assistant<|end_header_id|>\n\n'
    return pmpt

# def gen_C(x, ls):
#     C = [[ls[0]]]
#     for i in ls:
#         cl = False
#         if i != ls[0]:
#             for c in C:
#                 if get_entailment(x, c[0], i) == 'entailment' and get_entailment(x, i, c[0]) == 'entailment':
#                     c.append(i);cl=True;break;
#         if cl==False:
#             C.append([i])
# #    return C

def gen_prob(problem ,prompt, num=1, verify=False, include_eg = True):
    responses = []
    tokenlist = []
    problist = []
    max_tokens = 1024

    # print(prompt, problem, )
    
    for i in range(num):
        ans_valid = False
        string_y = ''
        logitz = []
        tokens = []
        while not ans_valid:
            logitz = []
            tokens = []
            string_y = ''
            # score 30 samples with humans to check correlation.
            
            msg = gen_chat_object(prompt, problem, include_eg=include_eg)  
#             print("MSG: " + msg)
            inputs = tokenizer(
                [
                msg
                ], return_tensors = "pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, use_cache=True, output_logits = True, return_dict_in_generate = True, )
            output_logits = outputs.logits
            # creates token list 
            for i in range(len(outputs.sequences[0]) - 1): # leave out EOS token
                item = outputs.sequences[0][i]
                tokens.append(tokenizer.decode(item))
            # removes prompt 
            tokens = tokens[-len(output_logits):]
            # creates string 
            for token in tokens:
                string_y += token
            print("STRING: ", string_y)
            # gets logits index
            logitindices = outputs.sequences[0][-len(output_logits):]
            
            for i in range(len(tokens) - 1):
                probs = torch.nn.functional.log_softmax(output_logits[i], dim=1)
#                 print(probs[0][logitindices[i].item()])
                logitz.append(probs[0][logitindices[i].item()].item())
            if string_y.count("Step") + string_y.count("step") == 1 or verify == False:
                ans_valid = True
            elif "STOP" in string_y:
                ans_valid = True
            else:
                print("REGENERATING, STEP ERROR")
        
        problist.append(logitz)
        tokenlist.append(tokens)
        responses.append(string_y)
    # print(responses)
    return responses, tokenlist, problist
    
# output = gen_prob(macgyver[0]['Problem'], prompt=prompt, num=5)
# print(output)