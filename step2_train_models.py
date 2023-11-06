# %% even more imports
import os

from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast, AutoTokenizer, DataCollatorForTokenClassification

# %% read BILU HuggingFace dataset from disk
bilus_hug = Dataset.load_from_disk(dataset_path=os.path.join('data', 'BILUs_hf'))
print(bilus_hug)
print(bilus_hug.features)

# %% split dataset into train test val
train_testvalid = bilus_hug.train_test_split(test_size = 0.2, seed = 42)
test_valid = train_testvalid['test'].train_test_split(test_size = 0.5, seed = 42)
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


def batch_tokenise_and_embed(batch):
    # tokenise
    tokenised_inputs = tokeniser(batch['Text'], truncation=True)
    # # align annotation with added [CLS] and [SEP]
    # for bilu_column in ['EVENT-BILUs','LOC-BILUs','MISC-BILUs','ORG-BILUs','PER-BILUs','TIME-BILUs']:
    #     all_labels = batch[bilu_column]
    #     new_labels = [[-100, *labels, -100] for labels in all_labels]
    #     tokenised_inputs[bilu_column] = new_labels
    return tokenised_inputs


bilus_hug_tokenised = bilus_hug.map(
    batch_tokenise_and_embed,
    batched=True,
    remove_columns=bilus_hug["train"].column_names
)
print(bilus_hug_tokenised)

# %% get a sample
sample = bilus_hug_tokenised["train"][1]
sample

# %% training pipeline
data_collator = DataCollatorForTokenClassification(tokenizer=tokeniser, padding=True)
batch = data_collator([bilus_hug_tokenised["train"][i] for i in range(2)])
batch

# %% debug
pass
