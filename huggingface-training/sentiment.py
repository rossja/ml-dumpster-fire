import warnings
warnings.filterwarnings("ignore", category=Warning)

def classify(input_text):
  from transformers import pipeline
  classifier = pipeline("text-classification",
    model='bhadresh-savani/distilbert-base-uncased-emotion',
    top_k=None
  )
  result = classifier(input_text)
  return result

text = "I love using Hugging Face for NLP tasks!"
result = classify(text)
print(result)