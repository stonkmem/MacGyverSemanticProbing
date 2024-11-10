from transformers import pipeline

pipe = pipeline("text-classification",model="tasksource/deberta-base-long-nli")

def get_entailment_nli(a, b):
    return pipe({'text':a,
          'text_pair':b})['label']

def get_probs_nli(a, b):
    dic = pipe({'text':a,
          'text_pair':b})
    return (dic['label'], dic['score'])




# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# tokenizer = AutoTokenizer.from_pretrained("potsawee/deberta-v3-large-mnli")
# model = AutoModelForSequenceClassification.from_pretrained("potsawee/deberta-v3-large-mnli")

# def get_entailment_deberta(a,b):
#     textA = a
#     textB = b
#     entailed = "entailment"
#     contradict = "neutral"

#     inputs = tokenizer.batch_encode_plus(
#         batch_text_or_text_pairs=[(textA, textB)],
#         add_special_tokens=True, return_tensors="pt",
#     )
#     print(inputs)
#     logits = model(**inputs).logits 
#     probs = torch.softmax(logits, dim=-1)[0]
#     if probs[1] > 0.5:
#         return entailed
#     else:
#         return contradict

# get_entailment_deberta("Paris", "Capital of France")