from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-base')
tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-base')

features = tokenizer(['A man is eating pizza', 'A black race car starts up in front of a crowd of people.'], ['A man eats something', 'A man is driving down a lonely road.'],  padding=True, truncation=True, return_tensors="pt")

model.eval()
with torch.no_grad():
    scores = model(**features).logits
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
    print(labels)










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