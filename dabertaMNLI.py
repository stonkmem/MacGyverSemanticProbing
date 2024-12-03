from transformers import pipeline

pipe = pipeline("text-classification",model="tasksource/deberta-base-long-nli", device="cuda")

def get_entailment_nli(question, a, b):
    return pipe({'text':a,
          'text_pair':b})['label']

def get_probs_nli(question, a, b):
    dic = pipe({'text':a,
          'text_pair':b})
    return (dic['label'], dic['score'])

if __name__ == '__main__':
    print(get_probs_nli('Where is Paris?', 'Paris is in France.', "The country of France."))