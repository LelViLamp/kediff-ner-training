# %% even more imports
import os

import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer


# %% read BILU dataset and tokenise is
bilus_df = pd.read_csv(os.path.join("data", "BILUs.csv"))
bilus_hug = Dataset.from_pandas(bilus_df)
print(bilus_hug)


# %% split dataset into train test val
train_testvalid = bilus_hug.train_test_split(test_size = 0.2)
test_valid = train_testvalid['test'].train_test_split(test_size = 0.5)
# gather everyone if you want to have a single DatasetDict
bilus_hug = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']}
)
del train_testvalid, test_valid
print(bilus_hug)


# %% tokenisation
checkpoint: str = "dbmdz/bert-base-historic-multilingual-cased"
tokeniser: BertTokenizerFast = AutoTokenizer.from_pretrained(checkpoint)
print(f"Is '{checkpoint}' a fast tokeniser?", tokeniser.is_fast)


def batch_tokenise(batch):
    tokenised_inputs = tokeniser(batch['Text'], truncation=True)
    return tokenised_inputs


bilus_hug_tokenised = bilus_hug.map(batch_tokenise, batched=True)
print(bilus_hug_tokenised)


# %% debug
pass