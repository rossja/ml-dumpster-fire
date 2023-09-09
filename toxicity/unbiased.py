from detoxify import Detoxify
import pandas as pd

# each model takes in either a string or a list of strings
results = Detoxify('unbiased').predict(['white people are stupid'])

# optional to display results nicely
print(pd.DataFrame(results, index=input_text).round(5))
