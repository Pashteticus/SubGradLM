from datasets import load_dataset, DatasetDict, Features, Value
from utils.parsers_utils import *


instruction_features = Features(
    {
        "System": Value("string"),
        "User": Value("string"),
        "Assistant": Value("string"),
        "Language": Value("string"),
    }
)
reasonings_features = Features(
    {
        "System": Value("string"),
        "User": Value("string"),
        "Reasoning": Value("string"),
        "Assistant": Value("string"),
        "Language": Value("string"),
    }
)

with load_dataset(
    "csv", data_files=instructions_filename(), features=instruction_features
) as instructions_dataset:
    instructions_dataset.push_to_hub(REPO_PATH, config_name="instructions")

with load_dataset("csv", data_files=reasonings_filename()) as reasonings_dataset:
    reasonings_dataset.push_to_hub(REPO_PATH, config_name="reasonings_updated")
