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

def main():
  text = "I love using Hugging Face for NLP tasks!"
  results = classify(text)
  scores = {}
  for emotion in results[0]:
    scores.update({ emotion['label']: emotion['score']})
  sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
  # print(sorted_scores.pop(0))
  print(sorted_scores.pop(0)[0])

if __name__ == "__main__":
  main()