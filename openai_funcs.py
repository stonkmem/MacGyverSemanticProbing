
from openai import OpenAI
import os
# import openai

from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

entailment_llm_openai = OpenAI(api_key=api_key)

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
    llm_fact_openai = openai.OpenAI()
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

def get_factuality(question, ans, criteria):
    
    completion = client.chat.completions.create(
    model="gpt-4o",
        messages=[
                {'role':'system', 'content':'Please act as an impartial judge and evaluate the quality\
    of the solution provided by an AI assistant to the user problem displayed below. \
    Your evaluation should rate the feasability, efficiency, safety and effectiveness of the response. Be as objective as possible. \
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
print(get_entailment("Which country is New York in?", "USA", "New York is located in the USA.")) #['choices'][0]['message']['content']
# print(get_factuality(""))

def get_factuality_likert(question, ans, criteria):
    client=openai.OpenAI()
    completion = client.chat.completions.create(
    model="gpt-4o",
        messages=[
                {'role':'system', 'content':'Please act as an impartial judge and evaluate the quality\
    of the solution provided by an AI assistant to the user problem displayed below. \
    Your evaluation should rate the feasability, efficiency, safety and effectiveness of the response on a integer scale of 1 to 10. Be as objective as possible. \
    Do not provide explanation, and only state the score of the response in terms of efficiency, safety, effectiveness and feasiblity by strictly following this format: "Efficiency: [[<score>]], Safety: [[<score>]], Effectiveness: [[<score>]], Feasibility: [[<score>]]", for example: "Efficiency: [[5]], Safety: [[8]], Effectiveness: [[9]], Feasibility: [[2]]".'},
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

# %%
def gen_factuality_score(question, ans, criterialist):
    score = 0
    scores = get_factuality(question, ans, criteria)
    scores.split("[[")
    for i in len(scores):
        if i >= 1:
            if scores[i][0] == Y:
                score += 1
    if len(criterialist > 0):
        score /= len(criterialist)
        # finds avg
    else:
        score = -1
    return score

def gen_factuality_score_likert(question, ans, criterialist):
    score = 0
    scores = get_factuality_likert(question, ans, criteria)
    scores.split("[[")
    for i in len(scores):
        if i >= 1:
            try:
                if scores[i][1] == "0":
                    score += 1 # 10
                else:
                    score += int(scores[i][0]) / 10
            except:
                print("FORMAT ISSUE")
#     for criteria in criterialist:
#         factual = int(get_factuality(question, ans, criteria))
#         if factual > 0:
#             score += factual
    if len(criterialist)> 0:
        score /= len(criterialist) # normalise
        # finds avg
    else:
        score = -1
    return score

def gen_C(x, ls, tokenseq, probsq):
    C = [[ls[0]]]
    T = [[tokenseq[0]]]
    P = [[probsq[0]]]
    for i in ls:
        cl = False
        if i != ls[0]:
            for c in C:
              # break
#                 print(c[0], i)
#                 print(get_entailment(x, c[0], i))
                if (get_entailment(x, c[0], i) == 'entailment' and get_entailment(x, i, c[0]) == 'entailment') or i == c[0]:
                    c.append(i);
                    c_index = C.index(c)
                    T[c_index].append(tokenseq[ls.index(i)])
                    P[c_index].append(probsq[ls.index(i)])
                    print("c: ", c)
                    cl=True;break;
        if cl==False and i != ls[0]:
            C.append([i])
            T.append([tokenseq[ls.index(i)]])
            P.append([probsq[ls.index(i)]])
    return C, T, P

# %%
def generate_data_from_GPT(num_responses, inputs):
    responses = []
    problist = []
    tokenlist = []
    for i in range(num_responses):
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": inputs,
                }
            ],
            logprobs=True
        )
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

print(generate_data_from_GPT(1, "Write a haiku about recursion in programming."))