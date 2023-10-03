import warnings
warnings.filterwarnings("ignore", category=Warning)

def zero_shot_classification(text, candidate_labels):
  from transformers import pipeline
  classifier = pipeline("zero-shot-classification",
    model="facebook/bart-large-mnli")
  return classifier(text, candidate_labels)
  # return classifier(text, candidate_labels, multi_label=True)
  # using multi_label messes up the weights for some reason
  # without:
  # {'sequence': 'This is a course about the Transformers library', 'labels': ['education', 'business', 'politics'], 'scores': [0.8445988893508911, 0.11197422444820404, 0.04342687502503395]}
  # with:
  # {'sequence': 'This is a course about the Transformers library', 'labels': ['education', 'business', 'politics'], 'scores': [0.2589397132396698, 0.000499172427225858, 9.32908442337066e-05]}

def main():
  labels = ["education", "politics", "business"]
  text = "This is a course about the Transformers library"
  result = zero_shot_classification(text, labels)
  print(result)

if __name__ == "__main__":
  main()