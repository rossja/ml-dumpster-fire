import warnings
warnings.filterwarnings("ignore", category=Warning)

def zero_shot_classification(text, candidate_labels):
  from transformers import pipeline
  classifier = pipeline("zero-shot-classification",
    model="facebook/bart-large-mnli")
  return classifier(text, candidate_labels, , multi_label=True)

def main():
  labels = ["education", "politics", "business"]
  text = "This is a course about the Transformers library"
  result = zero_shot_classification(text, labels)
  print(result)

if __name__ == "__main__":
  main()