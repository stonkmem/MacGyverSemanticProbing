def get_entailment_openai(question, a, b):
    entailment_llm_openai = openai.OpenAI()
    return entailment_llm_openai.chat.completions.create(
        messages=[
            {'role': 'system', 'content': 'You are a skilled linguist studying semantic entailment. As a task, you will have to determine whether one sentence semantically entails another.'},
#             {'role': 'user', 'content': f'We are evaluating answers to the question: What disease does he have? .\n Here are 2 possible answers:\nSentence A: He has cancer\nSentence B: He is afflicted with cancer.\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, entailment, or neutral'},
#             {'role': 'assistant', 'content': 'entailment'},
            {'role': 'user', 'content': f'We are evaluating answers to the question: {question}.\n Here are 2 possible answers:\nSentence A: {a}\nSentence B: {b}\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, entailment, or neutral'}
        ],
        model='gpt-4o'
    )
# completion_with_backoff(
#         inst = entailment_llm_openai,
#         messages=[
#             {'role': 'system', 'content': 'You are a skilled linguist studying semantic entailment. As a task, you will have to determine whether one sentence semantically entails another.'},
# #             {'role': 'user', 'content': f'We are evaluating answers to the question: What disease does he have? .\n Here are 2 possible answers:\nSentence A: He has cancer\nSentence B: He is afflicted with cancer.\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, entailment, or neutral'},
# #             {'role': 'assistant', 'content': 'entailment'},
#             {'role': 'user', 'content': f'We are evaluating answers to the question: {question}.\n Here are 2 possible answers:\nSentence A: {a}\nSentence B: {b}\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, entailment, or neutral'}
#         ],
#         model='gpt-4o'
    # )
def get_corr_feas_eff_openai(fraege, antworten):
    ratings = []
    llm_fact_openai = openai.OpenAI()
    for i in range(len(fraege)):
        
        ratings.append(llm_fact_openai.chat.completions.create(messages=[
                {'role':'system', 'content':'Please act as an impartial judge and evaluate the quality\
    of the solution provided by an AI assistant to the user problem displayed below. \
    Your evaluation should rate the feasability and efficiency of the response. Be as objective as possible. \
    After providing your explanation, please state whether the response is or is not effective or feasible by strictly following this format: "Feasibility: [[YES/NO]], Efficiency: [[YES/NO]]", for example: "Feasibility: [[YES]], Efficiency: [[NO]]".'},
                {'role':'user','content':f"""[Question]
    {fraege[i]}
    
    [The Start of Assistant’s Answer]
    {antworten[i]}
    [The End of Assistant's Answer]"""}
            ]#['choices'][0]['message']['content']
        ))
        
    return ratings