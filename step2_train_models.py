# %% even more imports
import os

import evaluate
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast, AutoTokenizer, DataCollatorForTokenClassification

from helper import print_aligned

# %% read BILOU HuggingFace dataset from disk
BILOUs_hug = Dataset.load_from_disk(dataset_path=os.path.join('data', 'BILOUs_hf'))
print(BILOUs_hug)
print(BILOUs_hug.features)

# %% split dataset into train test val
train_testvalid = BILOUs_hug.train_test_split(test_size=0.2, seed=42)
test_valid = train_testvalid['test'].train_test_split(test_size=0.5, seed=42)
# gather everyone if you want to have a single DatasetDict
BILOUs_hug = DatasetDict({
    'train': train_testvalid['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']}
)
del train_testvalid, test_valid
print(BILOUs_hug)

# %% tokenisation
checkpoint: str = "dbmdz/bert-base-historic-multilingual-cased"
tokeniser: BertTokenizerFast = AutoTokenizer.from_pretrained(checkpoint)
print(f"Is '{checkpoint}' a fast tokeniser?", tokeniser.is_fast)


def batch_embed(batch):
    # align annotation with added [CLS] and [SEP]
    for BILOU_column in ['EVENT-BILOUs', 'LOC-BILOUs', 'MISC-BILOUs', 'ORG-BILOUs', 'PER-BILOUs', 'TIME-BILOUs']:
        all_labels = batch[BILOU_column]
        new_labels = [[-100, *labels[1:-1], -100] for labels in all_labels]
        batch[BILOU_column] = new_labels
    return batch


BILOUs_hug = BILOUs_hug.map(batch_embed, batched=True)


def batch_tokenise(batch):
    # tokenise
    tokenised_inputs = tokeniser(batch['Text'], truncation=True)
    tokenised_inputs["labels"] = batch['PER-BILOUs']
    return tokenised_inputs


BILOUs_hug_tokenised = BILOUs_hug.map(
    batch_tokenise,
    batched=True,
    remove_columns=BILOUs_hug["train"].column_names
)
print(BILOUs_hug_tokenised)

# %% get a sample
sample = BILOUs_hug_tokenised["train"][1]
sample

# %% data collation
data_collator = DataCollatorForTokenClassification(tokenizer=tokeniser, padding=True)
batch = data_collator([BILOUs_hug_tokenised["train"][i] for i in range(2)])
print(batch)
print(batch['labels'])

for i in range(2):
    print(BILOUs_hug_tokenised["train"][i]["labels"])

# %% scaffold a metric
metric = evaluate.load("seqeval")

label_names = BILOUs_hug["train"].features["PER-BILOUs"].feature.names

labels = BILOUs_hug["train"][1]["PER-BILOUs"]
labels = [label_names[i] for i in labels[1:-1]]

# fake predictions
predictions = labels.copy()
predictions[2] = "B-PER"
predictions[3] = "I-PER"

print_aligned(labels, predictions)
metric.compute(predictions=[predictions], references=[labels])

# %% debug
pass
