import json
from datasets import load_dataset

# Print all the available datasets
from huggingface_hub import list_datasets
datasets = [dataset.id for dataset in list_datasets()]
datasets.sort()
print(json.dumps(datasets))
