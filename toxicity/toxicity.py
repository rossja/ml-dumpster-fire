
# checks the toxicity of provided text
# returns a dictionary of probabilities for each toxicity category
def check_toxicity(text):
  from detoxify import Detoxify
  results = Detoxify("original").predict(text)
  return results

# runs a test sample of toxic and nontoxic text
def test():
  from detoxify import Detoxify

  # example toxic text
  toxic_text = "all of you are stupid and dumb"
  toxic_results = Detoxify("original").predict(toxic_text)
  # print("toxic: %s" % toxic_results)

  # example nontoxic text
  nontoxic_text = "all of you are smart and kind"
  nontoxic_results = Detoxify("original").predict(nontoxic_text)
  # print("nontoxic: %s" % nontoxic_results)

  # print the results formatted nicely
  print_results(toxic_results, toxic_text)
  print_results(nontoxic_results, nontoxic_text)

# display results nicely (will need to pip install pandas)
def print_results(results, text):
  import pandas as pd
  print("\n%s\n" % pd.DataFrame(results, index=[text]).round(5))

if __name__ == "__main__":
  test()