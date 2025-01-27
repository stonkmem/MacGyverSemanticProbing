# from llama_cpp import Llama
import sys
import os
import transformers
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel, BitsAndBytesConfig

from dotenv import load_dotenv
from huggingface_hub import InferenceClient, login

from src.helper_funcs import *
from dotenv import load_dotenv

load_dotenv()
temp = 1.0

if len(sys.argv) > 6:
    temp = float(sys.argv[6])
    print("TEMP: ", temp)

toggle_hs = False
if len(sys.argv) > 8:
    if sys.argv[8] == 'hs':
        toggle_hs = True
        print("HS ACTIVE")
    else:
        print("HS INACTIVE")
TOP_P = 0.9
NUM_BEAMS = 1

huggingface_token = os.getenv("HF_TOKEN")
print('HF_TOKEN' in os.environ) # True of False
# print(os.environ['HF_TOKEN']) # Print contents of variable
# login(token=huggingface_token, add_to_git_credential=True)
print(sys.argv, "ARGUMENTS")
os.environ["HF_HOME"] = "~/scratch/macgyversemanticprobing/.cache/"
os.environ["TRANSFORMERS_CACHE"] = "~/scratch/macgyversemanticprobing/.cache/"
# if __name__ == '__main__':
if len(sys.argv) < 2:
    modelpath = "meta-llama/Llama-3.1-8B-Instruct"
elif sys.argv[1] == 'llama' :
    print("LLAMA")
    modelpath = "meta-llama/Llama-3.1-8B-Instruct"
elif sys.argv[1] == 'llama30':
    print("llama 3.0")
    modelpath = "meta-llama/Meta-Llama-3-8B-Instruct"
elif sys.argv[1] == 'llama2':
    print('llama2')
    modelpath = "meta-llama/Llama-2-70b-hf" # meta-llama/Llama-2-7b-hf
elif sys.argv[1] == 'llama3-70b':
    print('llama3-70b')
    modelpath = "meta-llama/Meta-Llama-3-70B-Instruct"
elif sys.argv[1] == "llama3.2":
    print("llama3.2")
    modelpath = "meta-llama/Llama-3.2-3B-Instruct"
elif sys.argv[1] == 'llama3.21b':
    print('llama3.2 1b')
    modelpath = "meta-llama/Llama-3.2-1B-Instruct"
elif sys.argv[1] == 'vicuna':
    modelpath = "lmsys/vicuna-13b-v1.5"
    print("VICUNA")
elif sys.argv[1] == 'mistral':
    modelpath = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    print("MISTRAL")
elif sys.argv[1] == 'llama_70b':
    modelpath = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    print("LLAMA 70B")
elif sys.argv[1] == 'llama3.3':
    modelpath = "meta-llama/Llama-3.3-70B-Instruct"
    print("LLAMA 3.3 70B")
elif sys.argv[1] == 'vicuna-7b':
    modelpath = "lmsys/vicuna-7b-v1.5"
    print("VICUNA 7B")
elif sys.argv[1] == 'vicuna-33b':
    modelpath = "lmsys/vicuna-33b-v1.3"
    print("VICUNA 13B")
elif sys.argv[1] == "mistral-nemo":
    modelpath = "mistralai/Mistral-Nemo-Instruct-2407"
    print("MISTRAL NEMO")
elif sys.argv[1] == "mistral-large":
    modelpath = "mistralai/Mistral-Large-Instruct-2407"
    print("MISTRAL LARGE")
elif sys.argv[1] == "ministral":
    modelpath = "mistralai/Ministral-8B-Instruct-2410"
    print("MINISTRAL")
else:
    modelpath = "meta-llama/Llama-3.1-8B-Instruct"
    print("OTHER")
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
if modelpath == "meta-llama/Llama-2-70b-hf" or modelpath == "meta-llama/Meta-Llama-3-70B-Instruct":
    tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False, add_bos_token = True, legacy=False) 
else:
    tokenizer = AutoTokenizer.from_pretrained(modelpath, use_fast = False, add_bos_token = False, legacy=False)

if modelpath == "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF" or modelpath == "meta-llama/Llama-3.3-70B-Instruct" or modelpath == 'meta-llama/Llama-2-70b-hf' or modelpath == 'meta-llama/Meta-Llama-3-70B-Instruct':
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(modelpath, device_map = 'auto', torch_dtype=torch.bfloat16, quantization_config=quantization_config)
elif modelpath != "gpt4":
    model = AutoModelForCausalLM.from_pretrained(modelpath, device_map = 'auto')
# model = AutoModelForCausalLM.from_pretrained(modelpath, device_map = 'auto')
print("MODEL LOADED")
now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
print(torch.cuda.get_device_properties(0).total_memory, "TOTAL MEMORY")


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

max_count = 5

def gen_prob(problem ,prompt, num=1, verify=False, include_eg = True):
    responses = []
    tokenlist = []
    problist = []
    max_tokens = 1024
    hiddenstates = []
    msg = gen_chat_object(prompt, problem, include_eg=include_eg)  
    print("MSG: " , msg)
    inputs = tokenizer(
        
        msg
        , return_tensors = "pt").to("cuda")
    for i in range(num):
        ans_valid = False
        string_y = ''
        logitz = []
        tokens = []
        # max_count = 10
        counter = 0
        
        while not ans_valid:
            logitz = []
            tokens = []
            string_y = ''
            if toggle_hs:
                outputs = model.generate(**inputs, max_new_tokens=max_tokens, use_cache=True, output_logits = True, return_dict_in_generate = True, 
                                        temperature=temp,
                                    top_p = TOP_P, do_sample = True,
                                    output_hidden_states = True,
                                        num_beams = NUM_BEAMS)
            else:
                outputs = model.generate(**inputs, max_new_tokens=max_tokens, use_cache=True, output_logits = True, return_dict_in_generate = True, 
                                        temperature=temp,
                                    top_p = TOP_P, do_sample = True,
                                        num_beams = NUM_BEAMS)
            output_logits = outputs.logits
            if toggle_hs:
                hidden_states = outputs.hidden_states
            else:
                hidden_states = []
            # creates token list 
            for i in range(len(outputs.sequences[0]) - 1): # leave out EOS token
                item = outputs.sequences[0][i]
                tokens.append(tokenizer.decode(item))
            # removes prompt 
            tokens = tokens[-len(output_logits):]
            # creates string 
            for token in tokens:
                string_y += token
            print("OUTPUT_STRING", string_y)
            string_y = string_y.replace(tokenizer.eos_token, "")
            # gets logits index
            logitindices = outputs.sequences[0][-len(output_logits):]
            
            counter += 1
            for i in range(len(tokens) - 1):
                probs = torch.nn.functional.log_softmax(output_logits[i], dim=1)
#                 print(probs[0][logitindices[i].item()])
                logitz.append(probs[0][logitindices[i].item()].item())
            if string_y.count("Step ") <= 2 or verify == False or counter > max_count:
                ans_valid = True
            elif "STOP" in string_y:
                ans_valid = True
            else:
                print("REGENERATING, STEP ERROR", string_y)
        
        problist.append(logitz)
        tokenlist.append(tokens)
        responses.append(string_y)
        selected_indices =  [0, 8, 16, 24, 32] #  # 
        if modelpath == "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF":
            selected_indices = [0, 16, 32, 48, 64, 80]
        if len(hidden_states) > 1 and toggle_hs:
            second_last_hs = hidden_states[-2]
            # print(len(second_last_hs), "SECOND LAST HS")
            selected_tensors = [second_last_hs[idx] for idx in selected_indices]
            detensored_hs = [t.tolist() for t in selected_tensors]
            # detensored_hs = hidden_states[-2][-1].tolist()
            # for i in range(len(hidden_states[-1])): # remove tensors
            #     detensored_hs.append(hidden_states[-1][i].tolist())
            hiddenstates.append(detensored_hs)
            detensored_hs = []
            for hiddenstate in hidden_states:
                hiddenstate = [t.detach().cpu() for t in hiddenstate]
            selected_tensors = [t.detach().cpu() for t in selected_tensors]
            hidden_states = []
            selected_tensors = []
        else:
            hiddenstates.append([])
        # hiddenstates.append(hidden_states[-1][-1].tolist())
    # print(responses)

    # print(responses)
    return responses, tokenlist, problist, hiddenstates
    
# output = gen_prob(macgyver[0]['Problem'], prompt=prompt, num=5)
# print(output)

def gen_prob_mistral(problem ,prompt, num=1, verify=False, include_eg = True):
    responses = []
    tokenlist = []
    problist = []
    hiddenstates = []
    max_tokens = 1024

    # print(prompt, problem, )
    
    for i in range(num):
        ans_valid = False
        string_y = ''
        logitz = []
        tokens = []
        msg = gen_chat_object_mistral(prompt, problem, include_eg=include_eg)  

        encodeds = tokenizer.apply_chat_template(msg, tokenize=False, ) # add_generation_prompt=True
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(encodeds, return_tensors="pt", padding=True).to("cuda")

        # max_count = 10
        counter = 0
        while not ans_valid:
            logitz = []
            tokens = []
            string_y = ''
            # score 30 samples with humans to check correlation.
            
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, use_cache=True, output_logits = True, return_dict_in_generate = True,
                                     temperature=temp,
                                top_p = TOP_P, do_sample = True,
                                output_hidden_states = True,
                                    num_beams = NUM_BEAMS)
            output_logits = outputs.logits
            hidden_states = outputs.hidden_states
            tokens_previous = outputs.sequences[0]
            token_text = tokenizer.decode(tokens_previous)
            # print(outputs)
            # creates token list 
            for i in range(len(outputs.sequences[0]) - 1): # leave out EOS token and INST token
                item = outputs.sequences[0][i]
                tokens.append(tokenizer.decode(item))
            # removes prompt 
            tokens = tokens[-len(output_logits):]
            
            # creates string 
            str_index = token_text.index("[/INST]") + 7
            string_y = token_text[str_index:]
            string_y = string_y.replace("[/INST]", "")
            string_y = string_y.replace(tokenizer.eos_token, "")
            
            # gets logits index
            logitindices = outputs.sequences[0][-len(output_logits):]
            
            for i in range(len(tokens)):
                probs = torch.nn.functional.log_softmax(output_logits[i], dim=1)
#                 print(probs[0][logitindices[i].item()])
                logitz.append(probs[0][logitindices[i].item()].item())
            tokens = tokens[1:]
            logitz = logitz[1:]
            # print("STRING: ", string_y)
            # print("TOKENS: ", tokens)
            counter += 1
            if string_y.count("Step ") <= 2 or verify == False or counter > max_count:
                ans_valid = True
            elif "STOP" in string_y:
                ans_valid = True
            else:
                print("REGENERATING, STEP ERROR", string_y)
        
        problist.append(logitz)
        tokenlist.append(tokens)
        responses.append(string_y)
        # detensored_hs = []
        # for i in range(len(hidden_states[-1])):
        #     detensored_hs.append(hidden_states[-1][i].tolist())
        
        # hiddenstates.append(hidden_states[-1][-1].tolist())
    # print(responses)
    return responses, tokenlist, problist, hiddenstates

def gen_prob_vicuna(problem ,prompt, num=1, verify=False, include_eg = True):
    responses = []
    tokenlist = []
    problist = []
    hiddenstates = []
    max_tokens = 1024

    # print(prompt, problem, )
    msg = gen_chat_object(prompt, problem, include_eg=include_eg)  
    # tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer([msg], return_tensors="pt", padding=True).to("cuda")
    for i in range(num):
        ans_valid = False
        string_y = ''
        logitz = []
        tokens = []
        
        # max_count = 10
        counter = 0
        while not ans_valid:
            logitz = []
            tokens = []
            string_y = ''
            # score 30 samples with humans to check correlation.
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, use_cache=True, output_logits = True, return_dict_in_generate = True,
                                     temperature=temp,
                                top_p = TOP_P, do_sample = True,
                                # output_hidden_states = True,
                                    num_beams = NUM_BEAMS)
            output_logits = outputs.logits
            # hidden_states = outputs.hidden_states
            # tokens_previous = outputs.sequences[0]
#             tokens_previous = torch.cat((tokens_previous, input_ids), dim=1) # consider tokens_previous already generated tokens
            # full_token_text = tokenizer.decode(tokens_previous)
            # token_text  =  full_token_text # consider previous_output_length the length of the previous full_token_text
            # creates token list 
            items = []
            for i in range(len(outputs.sequences[0]) - 1): # leave out EOS token and INST token
                item = outputs.sequences[0][i]
                items.append(item)
                tokens.append(tokenizer.decode(item))
            # removes prompt 
            tokens = tokens[-len(output_logits):]
            items = items[-len(output_logits):]
            
            # creates string 
            string_y = tokenizer.decode(items, skip_special_tokens=True)
            string_y = string_y.replace(tokenizer.eos_token, "")
            string_y = string_y.replace(tokenizer.pad_token, "")
            string_y = string_y.replace("Response: ", "")
            
            # gets logits index
            logitindices = outputs.sequences[0][-len(output_logits):]
            
            for i in range(len(tokens)):
                probs = torch.nn.functional.log_softmax(output_logits[i], dim=1)
                logitz.append(probs[0][logitindices[i].item()].item())
            tokens = tokens[1:]
            logitz = logitz[1:]
            counter += 1
            if string_y.count("Step ") <= 2 or verify == False or counter > max_count:
                ans_valid = True
            elif "STOP" in string_y:
                ans_valid = True
            else:
                print("REGENERATING, STEP ERROR", string_y)
        
        problist.append(logitz)
        tokenlist.append(tokens)
        responses.append(string_y)
    return responses, tokenlist, problist, hiddenstates
