 

from openai_funcs import gen_C, gen_chat_object_GPT, generate_data_from_GPT, get_entailment_openai, get_entailment, get_factuality, get_factuality_chateval_binary, get_factuality_chateval_likert, get_factuality_likert, get_corr_feas_eff_openai
# from llama_funcs import model, tokenizer
from data import *
# from process_data import *
import numpy as np
import torch

print("RUNNING HELPER FUNCS")
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
def generate_tokens_and_probabilities(inputs, max_tokens=512):
    # Tokenize the prompt
    # inp/uts = tokenizer(prompt, return_tensors="pt").to(device)

    # Prepare list to store tokens and their respective probabilities
    token_list = []
    prob_list = []

    # Prepare a string to store the final response
    full_response = ""
    device = "cuda"

    # Perform token generation step-by-step, keeping track of probabilities
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, use_cache=True, output_logits = True, return_dict_in_generate = True)
        output_logits = outputs.logits
        full_response = tokenizer.batch_decode(outputs.sequences)[0]

        for i in range(len(output_logits)):
            # Get the last token logits (for the current step in generation)
            # next_token_logits = output_logits[:, -1, :]
            next_token_logits = output_logits[i]

            # Softmax to get probabilities
            # next_token_probs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
            next_token_probs = torch.nn.functional.softmax(next_token_logits, dim=-1)

            # Get the predicted token ID and its probability
            predicted_token_id = torch.argmax(next_token_probs, dim=-1).item()
            predicted_token_prob = next_token_probs[0, predicted_token_id].item()

            # Decode the token and store its probability
            decoded_token = tokenizer.decode([predicted_token_id])
            token_list.append(decoded_token)
            prob_list.append(predicted_token_prob)

            # Add the predicted token to the input for the next iteration
            # inputs["input_ids"] = torch.cat([inputs["input_ids"], torch.tensor([[predicted_token_id]]).to(device)], dim=1)

            # Concatenate the generated token to the final response string
            # full_response += decoded_token

            # Update output logits with new input
            # output_logits =
            # output_logits = unsloth(inputs["input_ids"], return_dict=True).logits

    return full_response, token_list, prob_list


 
import re

def preprocess_sequence(text, sequence):
    # Use regex to replace consecutive occurrences of the sequence with a single occurrence
    pattern = f'({re.escape(sequence)})\\s*\\1+'
    return re.sub(pattern, r'\1', text)

def split_by_sequence(text, sequence):

    cleaned_text = preprocess_sequence(text, sequence)

    # Use regex to split the text at occurrences of the sequence, but keeping the delimiter
    parts = re.split(f'({re.escape(sequence)})', cleaned_text)

    # Reassemble so that each split part includes the starting sequence
    result = []
    for i in range(1, len(parts), 2):  # We go in steps of 2 to skip over the plain text parts
        result.append(parts[i] + parts[i + 1])

    return result

# Example usage
input_text = """Step 1: Do this.
Some more text for step 1.
Step 1: Do that.
Additional details.
Step 1: Finish up.
Final remarks."""

# Split the text based on "Step 1:"
split_strings = split_by_sequence(input_text, "Step 1:")

# Printing each split part
for i, split_str in enumerate(split_strings, start=1):
    print(f"--- Split {i} ---\n{split_str}\n")


 
import math

def calculate_sequence_probability(probabilities, use_log_prob=False):
    """
    Calculate the overall probability of generating a token sequence
    given the individual token probabilities.

    Args:
        probabilities (list): A list of probabilities of individual tokens.
        use_log_prob (bool): If True, calculate using log probabilities to prevent underflow.

    Returns:
        float: The overall probability of generating the sequence.
    """

    if use_log_prob:
        # Calculate the log-probability sum to avoid numerical underflow

        log_prob_sum = sum(math.log(p) for p in probabilities if p > 0)
        print(log_prob_sum)
        return math.exp(log_prob_sum)
    else:
        # Direct product of probabilities
        overall_prob = 1.0
        for prob in probabilities:
            overall_prob *= prob
        return overall_prob

# Example usage:
probabilities = [0.9, 0.8, 0.85, 0.95]  # Example token probabilities
overall_probability = calculate_sequence_probability(probabilities)
print(f"Overall Probability: {overall_probability}")

# If you want to avoid numerical underflow with log-probabilities:
overall_probability_log = calculate_sequence_probability(probabilities, use_log_prob=True)
print(f"Overall Probability (using log-probs): {overall_probability_log}")


 
def preprocess_token_sequence(tokens, token_sequences):
    """
    Remove consecutive duplicate occurrences of any token sequence in the list.

    Args:
    - tokens: List of tokens (e.g., ["Step", " ", "1", ":", "Step", " ", "1", ":"])
    - token_sequences: List of token sequences to check for consecutive duplicates (e.g., [["Step", " ", "1", ":"]])

    Returns:
    - A list of tokens with consecutive duplicates of the sequences removed.
    """
    cleaned_tokens = []
    i = 0

    while i < len(tokens):
        found_duplicate = False

        for seq in token_sequences:
            seq_len = len(seq)

            # Check if current position matches the sequence exactly
            if tokens[i:i + seq_len] == seq:
                # Add the sequence once to the cleaned_tokens
                cleaned_tokens.extend(seq)

                # Skip over consecutive duplicates of the same sequence
                while tokens[i:i + seq_len] == seq:
                    i += seq_len  # Skip the entire sequence, not individual tokens

                found_duplicate = True
                break

        if not found_duplicate:
            # If no sequence matches, just append the current token
            cleaned_tokens.append(tokens[i])
            i += 1

    return cleaned_tokens


# Example usage with tokens containing spaces
tokens = ["Step", " ", "1", ":", "Step", " ", "1", ":", "Step", " ", "1", ":", "Some", "text", "Step", " ", "1", ":"]
sequences = [["Step", " ", "1", ":"]]  # The sequences to remove duplicates for

# Clean the tokens list
cleaned_tokens = preprocess_token_sequence(tokens, sequences)

print("Cleaned Tokens:", cleaned_tokens)


 
def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

import random

 
def generate_data(num_responses, inputs):
  responses = []
  problist = []
  tokenlist = []
  for i in range(num_responses):
    # inputs = tokenizer(
    #   [
    #   inputstring
    #   ]
    #   , return_tensors = "pt").to("cuda")
    response, token, prob = generate_tokens_and_probabilities(inputs)
    # response_index = response.index("Step 1: ")
    # responses.append(response[response_index:])
    if token[-1] == "<|eot_id|>":
      response = response[:-len(token[-1])]
      token = token[:-1]
      prob = prob[:-1]


    problist.append(prob)
    tokenlist.append(token)
    responses.append(response)
  return responses, tokenlist, problist

 
# completion = client.chat.completions.create(
#     model="gpt-4o-mini",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {
#             "role": "user",
#             "content": "Write a haiku about recursion in programming."
#         }
#     ],
#     logprobs=True
    
# )

# print(completion.choices[0].logprobs.content)



 
# def generate_data_from_LlamaCPP(num_responses, inputs):
#     responses = []
#     problist = []
#     tokenlist = []
#     for i in range(num_responses):
#         completion = entailment_llm.create_chat_completion(
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {
#                     "role": "user",
#                     "content": inputs,
#                 }
#             ],
#             logprobs=True,
#         )
#         print(completion['choices'][0][''])
#         # responses.append(completion.choices[0].message.content)
#         # tokens = []
#         # probs = []
#         # # print(len(completion.choices[0].logprobs.content))
#         # for j in range(len(completion.choices[0].logprobs.content)):
#         #     tokens.append(completion.choices[0].logprobs.content[j].token)
#         #     probs.append(completion.choices[0].logprobs.content[j].logprob)
#         #     # NOTE THAT IT IS ALREADY IN LOGPROB FORM
#         # tokenlist.append(tokens)
#         # problist.append(probs)
#     return responses, tokenlist, problist

# print(generate_data_from_LlamaCPP(1, "Write a haiku about recursion in programming."))


import math
def calc_sequence_probability_LOGPROB(probabilities, return_logprob = False):
    # print(probabilities)
    logprob = sum(probabilities)
    if return_logprob == False:
        return math.exp(logprob)
    else:
        return logprob

# print(calc_sequence_probability_LOGPROB(generate_data_from_GPT(1, "Write a haiku about recursion in programming.")[2][0]))


def extract_problem(prompt):
  problem_index = prompt.index("Problem:")
  problem = prompt[problem_index:]
  return problem


num_to_string = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten"
}


def isolate_sub_responses(tokens, probabilities, sub_response_start):
  # issue: omits the first step.
    """
    Isolates sub-responses based on the starting phrase (sub_response_start)
    and stores them as token and probability sub-lists in a nested array.

    Args:
        tokens (list): A list of tokens generated by the model.
        probabilities (list): A list of probabilities corresponding to the tokens.
        sub_response_start (str): The phrase marking the start of each sub-response (e.g., "Step 1:").

    Returns:
        list: A nested list where each element is a tuple (sub_response_tokens, sub_response_probs)
              representing a sub-response and its corresponding probabilities.
    """

    print("Tokens: ", tokens)
    print("Probabilities: ", probabilities)
    print("Sub-response start: ", sub_response_start)
    # Initialize a list to hold sub-responses
    sub_responses = []

    # Temp lists to store tokens and probabilities of current sub-response
    current_tokens = []
    current_probabilities = []

    # A flag to check if we are within a sub-response
    in_sub_response = False

    # Concatenate tokens into a single string for easier search
    concatenated_tokens = "".join(tokens)

    # Check if the starting phrase is present in the response
    if sub_response_start not in concatenated_tokens:
        return "No sub-responses found for the given starting phrase."

    # Iterate over tokens and probabilities
    for i in range(len(tokens)):
        token = tokens[i]
        prob = probabilities[i]

        # Check if the current token is the start of a new sub-response
        concat_index_of_token = 0
        for n in range(i):
            concat_index_of_token += len(tokens[n])
        if concatenated_tokens.startswith(sub_response_start, concat_index_of_token):
            # If we're already in a sub-response, store the current one
            if in_sub_response:
                sub_responses.append((current_tokens, current_probabilities))
                current_tokens = []
                current_probabilities = []

            # Mark the start of a new sub-response
            in_sub_response = True

        # If we are in a sub-response, add the current token and probability
        if in_sub_response:
            current_tokens.append(token)
            current_probabilities.append(prob)

    # After the loop, append the last sub-response if any
    if in_sub_response:
        print("Appending last sub-response", current_tokens)
        sub_responses.append((current_tokens, current_probabilities))

    return sub_responses


def remove_duplicates(main_string, substring):
    """
    Remove everything after the second occurrence of a substring in a string, including the second occurrence itself.
    
    Args:
    - main_string: The main string from which everything after the second occurrence should be removed.
    - substring: The substring to find and remove the second occurrence of and everything after it.
    
    Returns:
    - The modified string with only the content before and including the first occurrence of the substring.
    """
    # Find the first occurrence of the substring
    first_occurrence_index = main_string.find(substring)
    
    # If the first occurrence is not found, return the original string
    if first_occurrence_index == -1:
        return main_string
    
    # Find the second occurrence of the substring, starting after the first
    second_occurrence_index = main_string.find(substring, first_occurrence_index + len(substring))
    
    # If the second occurrence is not found, return the string as it is (nothing to remove)
    if second_occurrence_index == -1:
        return main_string
    
    # Slice the string up to just before the second occurrence and return the result
    return main_string[:second_occurrence_index]

# # Example usage
# main_string = "Step 1: Do this. Step 1: Do that. Step 1: Finish up."
# substring = "Step 1:"

# result = remove_duplicates(main_string, substring)
# print(result)

def calculate_prob_of_class(class_sequences, sequence_probs): # list of list of probabilities of each token in a sequence given x
  # for each sequence there is an element (multiple sequences mean the diff versions of a step)
  # for each token in a sequence there is a probabilti -> second layer of nested array.
  prob = 0
  for i in range(len(class_sequences)):
    prob += calculate_sequence_probability(sequence_probs[i])
  return prob

def calculate_prob_of_class2(class_sequences_probs):
  prob = 0
  for i in range(len(class_sequences_probs)):
    prob += class_sequences_probs[i]
  return prob

def calculate_prob_of_class_logprobs(class_sequence_logprobs, return_logprob = False): # calculate directly from logprobs
    prob = 0
    if return_logprob == False:
        for i in range(len(class_sequence_logprobs)):
            # print(calc_sequence_probability_LOGPROB(class_sequence_logprobs[i]))
            prob += calc_sequence_probability_LOGPROB(class_sequence_logprobs[i]) 
            # array of probs from logprobs
    else:
        for i in range(len(class_sequence_logprobs)):
            prob += calc_sequence_probability_LOGPROB(class_sequence_logprobs[i], True) 
            # array of logprobs
    return prob

def calculate_SE_simple(class_probs):
  SE = 0
  for i in range(len(class_probs)):
    # print(class_probs[i])
    if class_probs[i] > 0:
        SE += class_probs[i] * math.log(class_probs[i])
  return -SE



def calculate_SE_complex(class_probs):
  SE = 0
  sum_classprobs = 0
  for j in range(len(class_probs)):
    sum_classprobs += class_probs[j]
  for i in range(len(class_probs)):
    # sum_classprobs = 0
    # for j in range(len(class_probs)):
      # sum_classprobs += class_probs[j]
    newprob = class_probs[i] / sum_classprobs
    if newprob > 0:
        SE += newprob * math.log(newprob)
  return -SE

def compute_total_score(SE, factuality):
  sum = 0
  for i in factuality:
    sum += SE * i
  return sum

def generate_problem_score_simple(total_scores):
  problem_score = 0
  for i in range(len(total_scores)):
    problem_score += total_scores[i]
  if len(total_scores) > 0:
      return problem_score / len(total_scores)
  else: return -1


priority_vector = [0.47295, 0.29784, 0.086711, 0.14250]
# feasibility, safety, efficiency, effectiveness 

def gen_factuality_score(question, ans, criterialist):
    score = 0
    scores = get_factuality(question, ans)
    arr = scores.split("[[")
    feasibility = True
    efficiency = True
    print(scores)
    for i in range(len(arr)):
        if i >= 1:
            if arr[i][0] == "Y":
                print("OK")
                score += 1
            if arr[i][0] == "N":
                if i == 2:
                    feasibility = False
                if i == 3:
                    efficiency = False
    if len(criterialist) > 0:
        score /= len(criterialist) # finds avg
    else:
        score = -1
    return score, feasibility, efficiency

def gen_factuality_score_likert(question, ans, criterialist): # element 2 is feas, 3 is effec

    # INPUT: question (str), ans (str), criterialist (list of str)
    # OUTPUT: score (float), feasibility (bool), efficiency (bool)
    # note: use priority_vector for weights
    # should also output the array 

    # safety should be separate and not in AHP 
    score = 0
    scores = get_factuality_likert(question, ans) # should return a string: [[<score>]], [[<score>]], [[<score>]], [[<score>]]
    # based on criteria 
    arr = scores.split("[[")
    scorearray = []
    while len(arr) <= 4:
        scores = get_factuality_likert(question, ans)
        arr = scores.split("[[")
    feasibility = True
    efficiency = True
    if len(arr) > 3:
        if int(arr[2][0]) <= 7 and int(arr[2][0]) < int(arr[3][0]) and arr[2][1] != "0":
            feasibility = False
        elif int(arr[3][0]) <= 7 and int(arr[3][0]) < int(arr[2][0]) and arr[3][1] != "0":
            efficiency = False
    for i in range(len(arr)):
        if i >= 1:
            try:
#                 print(arr[i][0], arr[i][1])
                if arr[i][1] == "0":
                    score += 1 * priority_vector[i - 1] # 10
                    scorearray.append(10)
                else:
                    score += int(arr[i][0]) / 10 * priority_vector[i - 1]
                    scorearray.append(int(arr[i][0]))
#                     if i == 2:
#                         if int(arr[2][0]) <= 6: # 7
#                             feasibility = False
#                     if i == 3:
#                         if int(arr[3][0]) <= 6:
#                             efficiency = False
            except:
                print("FORMAT ISSUE")
    if len(criterialist) <= 0:
        # score /= len(criterialist) # normalise
        score = -1
        # finds avg
    return score, feasibility, efficiency, scorearray

def gen_factuality_score_chateval_likert(question, ans, criterialist, priority_vector): # element 2 is feas, 3 is effec
    score = 0
    scores = get_factuality_chateval_likert(criterialist, question, ans) # should return a string: [[<score>]], [[<score>]], [[<score>]], [[<score>]]
    # based on criteria
    feasibility = True
    efficiency = True
    scorearray = []
    for i in range(len(criterialist)):
        criteria = criterialist[i]
        scores[i][criteria] = int(scores[i][criteria])
        if criteria == 'feasible':
            if scores[i][criteria] <= 7 and scores[i][criteria] < scores[criterialist.index("efficiency")]["efficiency"] and scores[i][criteria] != 0:
                feasibility = False
        elif criteria == 'efficient':
            if scores[i][criteria] <= 7 and scores[i][criteria] < scores[criterialist.index("feasibility")]["feasibility"] and scores[i][criteria] != 0:
                efficiency = False
        score += scores[i][criteria] / 10 * priority_vector[criteria]
        scorearray.append(scores[i][criteria])
    return score, feasibility, efficiency, scorearray

def lambda_return(scores, gamma, lambda_):
    """
    Computes the bootstrapped lambda returns for a given list of creativity scores.

    Args:
    scores (list): Creativity scores for each step in the solution.
    gamma (float): Discount factor (how much future creativity scores are discounted).
    lambda_ (float): Lambda for bootstrapped lambda returns (controls the mix between 1-step and multi-step returns).

    Returns:
    np.array: Lambda returns for each step.
    """
    n = len(scores)
    if n == 0:
        return -1
    returns = np.zeros(n)  # Array to store lambda returns for each step
    G = 0  # Initialize future return

    # Calculate lambda returns starting from the last step going backwards
    for t in reversed(range(n)):
        # Compute the bootstrapped lambda return for step t
        G = scores[t] + gamma * ((1 - lambda_) * scores[t] + lambda_ * G)
        returns[t] = G

    return returns

def total_lambda_score(scores, gamma=0.9, lambda_=0.8):
    """
    Computes the total creativity score of a problem using bootstrapped lambda returns.

    Args:
    scores (list): Creativity scores for each step in the solution.
    gamma (float): Discount factor.
    lambda_ (float): Lambda for bootstrapped lambda returns.

    Returns:
    float: The total creativity score for the problem.
    """
    lambda_returns = lambda_return(scores, gamma, lambda_)
    total_score = np.mean(lambda_returns)  # Aggregate score using mean or sum
    return total_score

def check_feasibility(feasibility):
    num_infeasibles = 0
    num_agreements = 0
    for i in range(len(feasibility)):
        if macgyver[i]['Label'] == 'infeasible':
#             print()
            if feasibility[i] == 0:
                num_agreements += 1
            num_infeasibles += 1
    print(num_infeasibles, num_agreements)
    if num_infeasibles == 0:
        return -1
    return num_agreements / num_infeasibles

def check_efficiency(efficiency):
    num_inefficients = 0
    num_agreements = 0
    for i in range(len(efficiency)):
        if macgyver[i]['Label'] == 'inefficient':
#             print()
            if efficiency[i] == 0:
                num_agreements += 1
            num_inefficients += 1
    print(num_inefficients, num_agreements)
    if num_inefficients == 0:
        return -1
    return num_agreements / num_inefficients


# def calc_sequence_probability_LOGPROB(probabilities, return_logprob = False):
#     print(probabilities)
#     logprob = sum(probabilities)
#     if return_logprob == False:
#         return math.exp(logprob)
#     else:
#         return logprob
    
# prompt = f"""Please act as Macgyver, an intelligent person skilled in using ordinary tools in unconventional ways to solve problems.
#     Given the problem below, create ONE possible next step {step_num} to a multi-stage solution considering all the constraints and previous steps, if any.
#     Solve the problem in the fewest steps possible.
#     Arrive at the complete solution by step {max_steps}, such that it can solve the problem.
#     Be clear, specific and concise, maintaining practicality.
#     Ensure that the step you generate brings you significantly closer to solving the problem fully.
    
#     Do not generate step {step_num + 1}, etc.
#     Do NOT include explanation, examples or any python (or other) code in your response.
#     Do NOT generate anything extra, and limit the length of the one step you generate to one sentence maximum.  
#     Make your response as creative and innovative as possible.

#     Respond STRICTLY in this format:
#     Step {step_num}: <generate version of step {step_num} here>

#     If a new step does not need to be generated to solve the problem, respond strictly with "STOP"
#     """

# prompt = f"""Please act as Macgyver, an intelligent person skilled in using ordinary tools in unconventional ways to solve problems.
#     Given the problem below, create ONE possible next step {step_num} to a multi-stage solution considering all the constraints and previous steps, if any.
#     Solve the problem in the fewest steps possible.
#     Arrive at the complete solution by step {max_steps}, such that it can solve the problem.
#     Be clear, specific and concise, maintaining practicality.
#     Ensure that the step you generate brings you significantly closer to solving the problem fully.
#     Make your response as creative and innovative as possible.
    
#     Do not generate step {step_num + 1}, etc.
#     Do NOT include explanation, examples or any python (or other) code in your response.
#     Do NOT generate anything extra, and limit the length of the one step you generate to one sentence maximum.  
#     If the existing steps can already solve the problem effectively, respond strictly with "STOP"

#     Respond STRICTLY in this format:
#     Step {step_num}: <generate version of step {step_num} here>
#     """

# def gen_chat_object(prompt, problem, include_eg = True):
#     example_problem = """Problem: You need to remove the pit from an avocado for your salad, but your utility knife is blunt. In your drawer, you find a stainless steel egg whisk, a wooden rolling pin, a wine bottle filled with Merlot, a heavy-duty garlic crusher, small plastic spoons, and a fork with prongs bent slightly inwards. Unfortunately, the only spoon available isn't sufficient for the task. How should you proceed to remove the pit?
#     Existing steps, if any: 
#     Response: """
#     example_step = "Step 1: Take the fork with the bent prongs and pierce the avocado in an outline around the pit."
#     if include_eg:
#         messages = [
#             {'role': 'system', 'content': prompt},
#                 {'role': 'user', 'content': prompt + example_problem},
#                 {'role': 'assistant', 'content': example_step}, # must add \n\n for end of assistant for LLAMA
#                 {'role': 'user', 'content': problem}
#         ]
#     else:
#         messages = [
#             {'role': 'system', 'content': prompt},
#             {'role': 'user', 'content': problem}
#         ]
#     return messages

def gen_chat_object(prompt, problem, include_eg = True): # for LLAMA and vicuna
    example_problem = """Problem: You need to remove the pit from an avocado for your salad, but your utility knife is blunt. In your drawer, you find a stainless steel egg whisk, a wooden rolling pin, a wine bottle filled with Merlot, a heavy-duty garlic crusher, small plastic spoons, and a fork with prongs bent slightly inwards. Unfortunately, the only spoon available isn't sufficient for the task. How should you proceed to remove the pit?
    Existing steps, if any: 
    Response: """
    example_step = "Step 1: Take the fork with the bent prongs and pierce the avocado in an outline around the pit."
    if include_eg:
        messages = prompt + '\n For example, an example problem and step could be: \n' + example_problem + '\n' + example_step + '\n\n Now, here is the problem you are given: \n Problem: \n' + problem 
        # [
        #     {'role': 'system', 'content': prompt},
        #         {'role': 'user', 'content': prompt + example_problem},
        #         {'role': 'assistant', 'content': example_step}, # must add \n\n for end of assistant for LLAMA
        #         {'role': 'user', 'content': problem}
        # ]
    else:
        messages = prompt + '\n### Problem: \n' + problem
        # = [
        #     {'role': 'system', 'content': prompt},
        #     {'role': 'user', 'content': problem}
        # ]
    return messages

def gen_chat_object_mistral(prompt, problem, include_eg = True): # for Mistral and GPT-4o
    example_problem = """Problem: You need to remove the pit from an avocado for your salad, but your utility knife is blunt. In your drawer, you find a stainless steel egg whisk, a wooden rolling pin, a wine bottle filled with Merlot, a heavy-duty garlic crusher, small plastic spoons, and a fork with prongs bent slightly inwards. Unfortunately, the only spoon available isn't sufficient for the task. How should you proceed to remove the pit?
    Existing steps, if any: 
    Response: """
    example_step = "Step 1: Take the fork with the bent prongs and pierce the avocado in an outline around the pit."
    if include_eg:
        messages =  [
            {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': prompt + example_problem},
                {'role': 'assistant', 'content': example_step}, # must add \n\n for end of assistant for LLAMA
                {'role': 'user', 'content': problem}
        ]
#         prompt + '\n For example, an example problem and step could be: \n' + example_problem + '\n' + example_step + '\n\n Now, here is the problem you are given: \n Problem: \n' + problem 
        #
    else:
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': problem}
        ]
#         prompt + '\n### Problem: \n' + problem
    return messages

# AUROC Score

def calculate_auroc_score(y_true, y_pred):
    """
    Calculate the Area Under the Receiver Operating Characteristic Curve (AUROC) score.

    Args:
    y_true (list): True labels (0 or 1) for each prediction.
    y_pred (list): Predicted probabilities for each prediction.

    Returns:
    float: The AUROC score.
    """
    auroc = roc_auc_score(y_true, y_pred)
    return auroc

import numpy as np
from sklearn.metrics import accuracy_score

def calculate_auarc(y_true, y_scores, threshold=0.5):

    # INPUTS: y_true (list), y_scores (list), threshold (float)
    # y_true: true binary labels (0 or 1)
    # y_scores: model's confidence scores for the positive class (probabilities)

    """
    Calculate the Area Under the Accuracy-Rejection Curve (AUARC) for binary classification.
    
    Parameters:
        y_true (array-like): True binary labels (0 or 1).
        y_scores (array-like): Model's confidence scores for the positive class.
        threshold (float): Decision threshold for classification (default: 0.5).
    
    Returns:
        float: AUARC score.
    """

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)   
    # Sort by confidence scores in descending order
    sorted_indices = np.argsort(-y_scores)
    y_true_sorted = np.array(y_true)[sorted_indices]
    y_scores_sorted = np.array(y_scores)[sorted_indices]
    
    # Initialize lists to store accuracy and rejection rates
    accuracies = []
    rejection_rates = []
    
    n = len(y_true_sorted)
    
    # Calculate accuracy at each rejection rate
    for i in range(n + 1):
        if i == 0:
            # No rejection, use all predictions
            y_pred = (y_scores_sorted >= threshold).astype(int)
        elif i < n:
            # Use predictions for the top (n - i) confident scores
            y_pred = (y_scores_sorted[:n - i] >= threshold).astype(int)
        else:
            # 100% rejection, set accuracy to 1.0 by convention
            accuracies.append(1.0)
            rejection_rates.append(1.0)
            break
        
        # Calculate accuracy for the remaining predictions
        if len(y_pred) > 0:
            accuracy = accuracy_score(y_true_sorted[:n - i], y_pred)
        else:
            accuracy = 1.0  # No predictions, so accuracy is defined as 1.0 by convention
        
        accuracies.append(accuracy)
        rejection_rates.append(i / n)
    
    # Calculate AUARC using the trapezoidal rule
    auarc = np.trapz(accuracies, rejection_rates)
    return auarc

# # Example usage
# y_true = np.array([0, 1, 1, 0, 1, 0, 1, 1, 0, 0])
# y_scores = np.array([0.1, 0.9, 0.8, 0.4, 0.95, 0.35, 0.7, 0.85, 0.2, 0.3])

# # Calculate AUARC
# auarc_score = calculate_auarc(y_true, y_scores)
# print("AUARC Score:", auarc_score)

def compute_total_score_2(SE, factuality):
    if len(SE) > 0:
        SE_avg = sum(SE) / len(SE)
    else:
        SE_avg = 0
    return factuality * SE_avg
