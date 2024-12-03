from openai import OpenAI
import os
from dotenv import load_dotenv
import sys
from dabertaMNLI import * 
# import openai
# from helper_funcs import *

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
use_gpt4 = True
if len(sys.argv) > 5: # for entailment 
    if sys.argv[5] == "false":
        use_gpt4 = False

if use_gpt4:

    client = OpenAI(api_key=api_key)

    entailment_llm_openai = OpenAI(api_key=api_key)
print("RUNNING OPENAI")



def get_entailment_openai(question, a, b):
    return entailment_llm_openai.chat.completions.create(
        messages=[
            {'role': 'system', 'content': 'You are a skilled linguist studying semantic entailment. As a task, you will have to determine whether one sentence semantically entails another.'},
            {'role': 'user', 'content': 'We are evaluating answers to the question: What disease does he have? .\n Here are 2 possible answers:\nSentence A: He has cancer\nSentence B: He is afflicted with cancer.\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, entailment, or neutral'},
            {'role': 'assistant', 'content': 'entailment'},
            {'role': 'user', 'content': f'We are evaluating answers to the question: {question}.\n Here are 2 possible answers:\nSentence A: {a}\nSentence B: {b}\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, entailment, or neutral'}
        ],
        model='gpt-4o'
    )

def get_corr_feas_eff_openai(fraege, antworten):
    ratings = []
    llm_fact_openai = client
    for i in range(len(fraege)):
        
        ratings.append(llm_fact_openai.chat.completions.create(messages=[
                {'role':'system', 'content':'Please act as an impartial judge and evaluate the quality\
    of the solution provided by an AI assistant to the user problem displayed below. \
    Your evaluation should rate the feasability (Is this action possible?), efficiency (Does this action make efficient use of resources?), safety (Can the action be performed safely without speical precautions?), and effectiveness (Does this really solve the problem?) of the response. Be as objective as possible. \
    After providing your explanation, please state whether the response is or is not effective or feasible by strictly following this format: "Feasibility: [[YES/NO]], Efficiency: [[YES/NO]], Safety: [[YES/NO]], Effectiveness: [[YES/NO]]", for example: "Feasibility: [[YES]], Efficiency: [[NO]], Safety [[YES]], Effectiveness: [[NO]]".'},
                {'role':'user','content':f"""[Question]
    {fraege[i]}
    
    [The Start of Assistant’s Answer]
    {antworten[i]}
    [The End of Assistant's Answer]"""}
            ]#['choices'][0]['message']['content']
        ))
        
    return ratings




def get_entailment(question, a, b):
    # use GPT-4o
    # relax strictness via prompt engineering 
#     entailment_llm.reset();
#     llm.reset();
    completion = client.chat.completions.create(
    model="gpt-4o",
        messages=[
            {'role': 'system', 'content': 'You are a skilled linguist sutdying semantic entailment. As a task, you will have to determine whether one sentence entails another.'},
            {'role': 'user', 'content': f'We are evaluating answers to the question: What disease does he have? .\n Here are 2 possible answers:\nSentence A: He has cancer\nSentence B: He is afflicted with cancer.\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, entailment, or neutral'},
            {'role': 'assistant', 'content': 'entailment'},
            {'role': 'user', 'content': f'We are evaluating answers to the question: {question}.\n Here are 2 possible answers:\nSentence A: {a}\nSentence B: {b}\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, "entailment", or "neutral"'}
        ]
    )
    return completion.choices[0].message.content
    
#     return entailment_llm.create_chat_completion(
#         messages=[
#             {'role': 'system', 'content': 'You are a skilled linguist studying semantic entailment. As a task, you will have to determine whether one sentence entails another.'},
#             {'role': 'user', 'content': f'We are evaluating answers to the question: What disease does he have? .\n Here are 2 possible answers:\nSentence A: He has cancer\nSentence B: He is afflicted with cancer.\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, entailment, or neutral'},
#             {'role': 'assistant', 'content': 'entailment'},
#             {'role': 'user', 'content': f'We are evaluating answers to the question: {question}.\n Here are 2 possible answers:\nSentence A: {a}\nSentence B: {b}\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, "entailment", or "neutral"'}
#         ]
#     )

def get_factuality(question, ans):
    
    completion = client.chat.completions.create(
    model="gpt-4o",
        messages=[
                {'role':'system', 'content':'Please act as an impartial judge and evaluate the quality\
    of the solution provided by an AI assistant to the user problem displayed below. \
    Your evaluation should rate the feasability, safety, efficiency and effectiveness of the response. Be as objective as possible. \
    Do not provide explanation, and only state whether the response is or is not efficient, safe, effective or feasible by strictly following this format: "Efficiency: [[YES/NO]], Safety: [[YES/NO]], Effectiveness: [[YES/NO]], Feasibility: [[YES/NO]]", for example: "Efficiency: [[YES]], Safety: [[NO]], Effectiveness: [[YES]], Feasibility: [[NO]]".'},
                {'role':'user','content':f"""[Question] {question} 
    [The Start of Assistant’s Answer]
    {ans}
    [The End of Assistant's Answer]"""}
            ]
#         messages=[
#             {'role': 'system', 'content': 'You are a skilled expert studying the factuality of solutions. As a task, you will have to determine the extent to which an answer to the problem fulfills a criteria.'},
#             {'role': 'user', 'content': f'We are evaluating answers to the question: How can someone reach a shelf 3 metres high?.\n Here is the possible answer:They can climb a ladder to reach it. \\n Is the answer provided feasible?\n Respond STRICTLY a one word answer, "yes", or "no"'},
#             {'role': 'assistant', 'content': 'yes'},
#             {'role': 'user', 'content': f'We are evaluating answers to the question: {question}.\n Here is the possible answer:{ans}\\n Is the answer provided {criteria}?\n Respond STRICTLY a one word answer, "yes", or "no"'}
#         ]
    )
    return completion.choices[0].message.content
# print(get_entailment("Which country is New York in?", "USA", "New York is located in the USA.")) #['choices'][0]['message']['content']
# print(get_factuality(""))

def get_factuality_likert(question, ans):
    # client=openai.OpenAI()
    completion = client.chat.completions.create(
    model="gpt-4o",
        messages=[
                {'role':'system', 'content':'Please act as an impartial judge and evaluate the quality\
    of the solution provided by an AI assistant to the user problem displayed below. \
    Your evaluation should rate the feasability, safety, efficiency and effectiveness of the response on a integer scale of 1 to 10. Be as objective as possible. \
    Do not provide explanation, and only state the score of the response in terms of efficiency, safety, effectiveness and feasiblity by strictly following this format: "Feasibility: [[<score>]], Safety: [[<score>]], Efficiency: [[<score>]], Effectiveness: [[<score>]]", for example: "Efficiency: [[5]], Safety: [[8]], Effectiveness: [[9]], Feasibility: [[2]]".'},
                {'role':'user','content':f"""[Question] {question} 
    [The Start of Assistant’s Answer]
    {ans}
    [The End of Assistant's Answer]"""}
            ]
#         messages=[
#             {'role': 'system', 'content': 'You are a skilled expert studying the factuality of solutions. As a task, you will have to determine the extent to which an answer to the problem fulfills a criteria.'},
#             {'role': 'user', 'content': f'We are evaluating answers to the question: How can someone reach a shelf 3 metres high?.\n Here is the possible answer:They can climb a ladder to reach it. \\n Is the answer provided feasible?\n Respond STRICTLY an integer score from 1 to 10.'},
#             {'role': 'assistant', 'content': '9'},
#             {'role': 'user', 'content': f'We are evaluating answers to the question: {question}.\n Here is the possible answer:{ans}\\n Is the answer provided {criteria}?\n Respond STRICTLY an integer score from 1 to 10.'}
#         ]
    )
    return completion.choices[0].message.content
      
def get_factuality_chateval_binary(crit, question, ans):
    rounds = 2
    evals = []
    for criterion in crit:
        hist = []
        for i in range(rounds):
            positive = {'role': 'system', 'content': f'You are a skilled expert, Debater 1, studying solutions to a problem. As a task, you will be provided with a problem, solution, and a criteria to judge it on. You are to produce a 50 word argument for how the solution meets the criterion of {criterion}.'}
            positivearr = [positive]
            positivearr.extend(hist)
            positivearr.append({'role':'user','content':f"""[Problem] {question} 
[The Start of Assistant’s Solution]
{ans}
[The End of Assistant's Solution]"""})
            completion_positive = client.chat.completions.create(
                model="gpt-4o",
                messages=positivearr
            ).choices[0].message.content
            hist.append({'role':'assistant', 'content':'I, Debater 1, argue that: ' + completion_positive})
            negative = {'role': 'system', 'content': f'You are a skilled expert, Debater 2, studying the solutions to a problem. As a task, you will be provided with a problem, solution, and a criteria to judge it on. You are to produce a 50 word argument for how the solution fails to meet the criterion of {criterion}.'}
            negativearr = [negative]
            negativearr.extend(hist)
            negativearr.append({'role':'user','content':f"""[Question] {question} 
[The Start of Assistant’s Answer]
{ans}
[The End of Assistant's Answer]"""})
            completion_negative = client.chat.completions.create(
                model="gpt-4o",
                messages=negativearr
            ).choices[0].message.content
            hist.append({'role':'assistant', 'content':'I, Debater 2, argue that: ' + completion_negative})
        juj = {'role': 'system', 'content': f'You are a wise judge studying the solutions to a problem. As a task, you will be provided with a transcript of a debate between two LLMs. Based on the arguments presented, conclude whether or not the solution to the problem fulfils the criterion of {criterion}.\n Present your answer STRICTLY as follows: {criterion}: [[YES/NO]]. For example, {criterion}: [[YES]]'}
        jujarr = [juj]
        jujarr.extend(hist)
        jujarr.append({'role':'user','content':f"""[Question] {question} 
[The Start of Assistant’s Answer]
{ans}
[The End of Assistant's Answer]"""})
        print(jujarr)
        output = client.chat.completions.create(
            model="gpt-4o",
            messages=jujarr
        ).choices[0].message.content
        print(output, "JUJ")
        evals.append({criterion : client.chat.completions.create(
            model="gpt-4o",
            messages=jujarr
        ).choices[0].message.content == "[[YES]]"})
    return evals #RETURNS an array comprising several dictionaries, each of which is in the following format: {'criterion', judgement}


def get_factuality_chateval_binary(crit, question, ans):
    rounds = 2
    evals = []
    for criterion in crit:
        hist = []
        for i in range(rounds):
            positive = {'role': 'system', 'content': f'You are a skilled expert, Debater 1, studying solutions to a problem. As a task, you will be provided with a problem, solution, and a criteria to judge it on. You are to produce a 50 word argument for how the solution meets the criterion of {criterion}.'}
            positivearr = [positive]
            positivearr.extend(hist)
            positivearr.append({'role':'user','content':f"""[Problem] {question} 
[The Start of Assistant’s Solution]
{ans}
[The End of Assistant's Solution]"""})
            completion_positive = client.chat.completions.create(
                model="gpt-4o",
                messages=positivearr
            ).choices[0].message.content
            hist.append({'role':'assistant', 'content':'I, Debater 1, argue that: ' + completion_positive})
            negative = {'role': 'system', 'content': f'You are a skilled expert, Debater 2, studying the solutions to a problem. As a task, you will be provided with a problem, solution, and a criteria to judge it on. You are to produce a 50 word argument for how the solution fails to meet the criterion of {criterion}.'}
            negativearr = [negative]
            negativearr.extend(hist)
            negativearr.append({'role':'user','content':f"""[Question] {question} 
[The Start of Assistant’s Answer]
{ans}
[The End of Assistant's Answer]"""})
            completion_negative = client.chat.completions.create(
                model="gpt-4o",
                messages=negativearr
            ).choices[0].message.content
            hist.append({'role':'assistant', 'content':'I, Debater 2, argue that: ' + completion_negative})
        juj = {'role': 'system', 'content': f'You are a wise judge studying the solutions to a problem. As a task, you will be provided with a transcript of a debate between two LLMs. Based on the arguments presented, conclude whether or not the solution to the problem fulfils the criterion of {criterion}.\n Present your answer STRICTLY as follows: {criterion}: [[YES/NO]]. For example, {criterion}: [[YES]]'}
        jujarr = [juj]
        jujarr.extend(hist)
        jujarr.append({'role':'user','content':f"""[Question] {question} 
[The Start of Assistant’s Answer]
{ans}
[The End of Assistant's Answer]"""})
        print(jujarr)
        output = client.chat.completions.create(
            model="gpt-4o",
            messages=jujarr
        ).choices[0].message.content
        # print(output, "JUJ")
        evals.append({criterion : client.chat.completions.create(
            model="gpt-4o",
            messages=jujarr
        ).choices[0].message.content == "[[YES]]"})
    return evals #RETURNS an array comprising several dictionaries, each of which is in the following format: {'criterion', judgement}

# print(get_factuality_chateval_binary(["safety", "effectiveness"], 'How do you cook an egg?', "fry it with a frying pan"))

def get_factuality_chateval_likert(crit, question, ans):
    rounds = 2
    evals = []
    for criterion in crit:
        hist = []
        for i in range(rounds):
            positive = {'role': 'system', 'content': f'You are a skilled expert, Debater 1, studying solutions to a problem. As a task, you will be provided with a problem, solution, and a criteria to judge it on. You are to produce a 50 word argument for how the solution meets the criterion of {criterion}.'}
            positivearr = [positive]
            positivearr.extend(hist)
            positivearr.append({'role':'user','content':f"""[Problem] {question} 
[The Start of Assistant’s Solution]
{ans}
[The End of Assistant's Solution]"""})
            completion_positive = client.chat.completions.create(
                model="gpt-4o",
                messages=positivearr
            ).choices[0].message.content
            hist.append({'role':'assistant', 'content':'I, Debater 1, argue that: ' + completion_positive})
            negative = {'role': 'system', 'content': f'You are a skilled expert, Debater 2, studying the solutions to a problem. As a task, you will be provided with a problem, solution, and a criteria to judge it on. You are to produce a 50 word argument for how the solution fails to meet the criterion of {criterion}.'}
            negativearr = [negative]
            negativearr.extend(hist)
            negativearr.append({'role':'user','content':f"""[Question] {question} 
[The Start of Assistant’s Answer]
{ans}
[The End of Assistant's Answer]"""})
            completion_negative = client.chat.completions.create(
                model="gpt-4o",
                messages=negativearr
            ).choices[0].message.content
            hist.append({'role':'assistant', 'content':'I, Debater 2, argue that: ' + completion_negative})
        juj = {'role': 'system', 'content': f'You are a wise judge studying the solutions to a problem. As a task, you will be provided with a transcript of a debate between two LLMs. Based on the arguments presented, conclude the extent to which a solution to the problem fulfils the criterion of {criterion}.\n Present your answer STRICTLY as an integer score of 1 to 10 as follows: {criterion}: [[<score>]]. For example, {criterion}: [[7]]'}
        jujarr = [juj]
        jujarr.extend(hist)
        jujarr.append({'role':'user','content':f"""[Question] {question} 
[The Start of Assistant’s Answer]
{ans}
[The End of Assistant's Answer]"""})
        # print(jujarr)
        output = client.chat.completions.create(
            model="gpt-4o",
            messages=jujarr
        ).choices[0].message.content
        
        score = 0
        try:
            index = output.index("[[")
            output = output[index + 2:]
            if output[1] == '0':
                score = 10
            else:
                score = output[0]
                # print(len(output))
        except:
            print('err0r')
        evals.append({criterion : score})
    return evals #RETURNS an array comprising several dictionaries, each of which is in the following format: {'criterion', judgement}

# print(get_factuality_chateval_likert(["safety", "effectiveness"], 'How do you cook an egg?', "fry it with a frying pan"))
# print(get_factuality_chateval_binary(["safety", "effectiveness"], 'How do you cook an egg?', "fry it with a frying pan"))

# def gen_factuality_score(question, ans, criterialist):
#     score = 0
#     scores = get_factuality(question, ans, criterialist)
#     scores.split("[[")
#     for i in len(scores):
#         if i >= 1:
#             if scores[i][0] == Y:
#                 score += 1
#     if len(criterialist > 0):
#         score /= len(criterialist)
#         # finds avg
#     else:
#         score = -1
#     return score

# def gen_factuality_score_likert(question, ans, criterialist):
#     score = 0
#     scores = get_factuality_likert(question, ans, criterialist)
#     scores.split("[[")
#     for i in len(scores):
#         if i >= 1:
#             try:
#                 if scores[i][1] == "0":
#                     score += 1 # 10
#                 else:
#                     score += int(scores[i][0]) / 10
#             except:
#                 print("FORMAT ISSUE")
# #     for criteria in criterialist:
# #         factual = int(get_factuality(question, ans, criteria))
# #         if factual > 0:
# #             score += factual
#     if len(criterialist)> 0:
#         score /= len(criterialist) # normalise
#         # finds avg
#     else:
#         score = -1
#       return score
def gen_C(x, ls, tokenseq, probsq):
    C = [[ls[0]]]
    T = [[tokenseq[0]]]
    P = [[probsq[0]]]
    # print(x, ls, probsq)
    for i in ls:
        cl = False
        index = ls.index(i)
        classindex = 0
        callstop = False
        for k in C:
            for j in k:
                if j == i:
                    callstop = True
                    break
            if not callstop:
                classindex += 1
        # classindex = C.index([i])
        if i != ls[0]:
            for c in C:
              # break
#                 print(c[0], i)
#                 print(get_entailment(x, c[0], i))
                if len(sys.argv) > 4:
                    if sys.argv[4] != "deberta":
                        if (get_entailment(x, c[0], i) == 'entailment' and get_entailment(x, i, c[0]) == 'entailment') or i == c[0]:
                            c.append(i);
                            c_index = C.index(c)
                            T[c_index].append(tokenseq[ls.index(i)])
                            P[c_index].append(probsq[ls.index(i)])
                            print("c: ", c)
                            cl=True;break;
                    else:
                        if (get_entailment_nli(x, c[0], i) == 'entailment' and get_entailment_nli(x, i, c[0]) == 'entailment') or i == c[0]:
                            c.append(i);
                            c_index = C.index(c)
                            T[c_index].append(tokenseq[ls.index(i)])
                            P[c_index].append(probsq[ls.index(i)])
                            print("c: ", c)
                            cl=True;break;
                else:
                    if (get_entailment(x, c[0], i) == 'entailment' and get_entailment(x, i, c[0]) == 'entailment') or i == c[0]:
                        c.append(i);
                        c_index = C.index(c)
                        T[c_index].append(tokenseq[ls.index(i)])
                        P[c_index].append(probsq[ls.index(i)])
                        print("c: ", c)
                        cl=True;break;
        elif index != 0 or ls.count(i) > 1:
            C[classindex].append(i);
            # c_index = C.index(c)
            T[classindex].append(tokenseq[ls.index(i)])
            P[classindex].append(probsq[ls.index(i)])
            # print("c: ", c)
        if cl==False and i != ls[0]:
            C.append([i])
            T.append([tokenseq[ls.index(i)]])
            P.append([probsq[ls.index(i)]])
    return C, T, P

def gen_chat_object_GPT(prompt, problem, include_eg = True):
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
    else:
        messages = [
            {'role': 'system', 'content': prompt},
            {'role': 'user', 'content': problem}
        ]
#         prompt + '\n### Problem: \n' + problem
    return messages

def generate_data_from_GPT(problem ,prompt, num=1, verify=False, include_eg = True):
    responses = []
    problist = []
    tokenlist = []
    max_tokens = 1024
    for i in range(num):
        ans_valid = False
        string_y = ''
        logitz = []
        tokens = []
        while not ans_valid:
            msg = gen_chat_object_GPT(prompt, problem, include_eg = include_eg)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=msg,
                logprobs=True
            )
            string_y = completion.choices[0].message.content
            if string_y.count("Step") + string_y.count("step") == 1 or verify == False:
                ans_valid = True
            elif "STOP" in string_y:
                ans_valid = True
            else:
                print("REGENERATING, STEP ERROR")
        
        responses.append(completion.choices[0].message.content)
        tokens = []
        probs = []
        # print(len(completion.choices[0].logprobs.content))
        for j in range(len(completion.choices[0].logprobs.content)):
            tokens.append(completion.choices[0].logprobs.content[j].token)
            probs.append(completion.choices[0].logprobs.content[j].logprob)
            # NOTE THAT IT IS ALREADY IN LOGPROB FORM
        tokenlist.append(tokens)
        problist.append(probs)
    return responses, tokenlist, problist
# def generate_data_from_GPT(num_responses, inputs):
#     responses = []
#     problist = []
#     tokenlist = []
#     for i in range(num_responses):
#         completion = client.chat.completions.create(
#             model="gpt-4o",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {
#                     "role": "user",
#                     "content": inputs,
#                 }
#             ],
#             logprobs=True
#         )
#         responses.append(completion.choices[0].message.content)
#         tokens = []
#         probs = []
#         # print(len(completion.choices[0].logprobs.content))
#         for j in range(len(completion.choices[0].logprobs.content)):
#             tokens.append(completion.choices[0].logprobs.content[j].token)
#             probs.append(completion.choices[0].logprobs.content[j].logprob)
#             # NOTE THAT IT IS ALREADY IN LOGPROB FORM
#         tokenlist.append(tokens)
#         problist.append(probs)
#     return responses, tokenlist, problist

# print(generate_data_from_GPT(1, "Write a haiku about recursion in programming."))