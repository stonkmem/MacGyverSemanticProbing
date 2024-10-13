from llama_cpp import Llama

def get_entailment(question, a, b):
    entailment_llm = Llama.from_pretrained(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename = 'Llama-3.2-3B-Instruct-Q6_K_L.gguf',
        logits_all = True,
        n_gpu_layers=-1
    )
    return entailment_llm.create_chat_completion(
        messages=[
            {'role': 'system', 'content': 'You are a skilled linguist studying semantic entailment. As a task, you will have to determine whether one sentence semantically entails another.'},
            {'role': 'user', 'content': f'We are evaluating answers to the question: {question}.\n Here are 2 possible answers:\nSentence A: {a}\nSentence B: {b}\nDoes Sentence A semantically entail Sentence B?\n Respond STRICTLY a one word answer, entailment, or neutral.'}
        ]
    )

def get_corr_feas_eff(fraege, antworten):
    llm_fact = Llama.from_pretrained(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename = 'Llama-3.2-3B-Instruct-Q6_K_L.gguf',
        logits_all = True,
        n_gpu_layers=-1
    )
    ratings = []
    for i in range(len(fraege)):
        
        ratings.append(llm_fact.create_chat_completion(messages=[
                {'role':'system', 'content':'Please act as an impartial judge and evaluate the quality\
of the solution provided by an AI assistant to the user problem displayed below. \
Your evaluation should rate the feasability and efficiency of the response. Be as objective as possible. \
After providing your explanation, please state whether the response is or is not effective or feasible by strictly following this format: "Feasibility: [[YES/NO]], Efficiency: [[YES/NO]]", for example: "Feasibility: [[YES]], Efficiency: [[NO]]".'},
                {'role':'user','content':f"""[Question]
    {fraege[i]}
    
    [The Start of Assistant’s Answer]
    {antworten[i]}
    [The End of Assistant's Answer]"""}
            ]['choices'][0]['message']['content']
        ))
    return ratings
