# %%
import os
import torch

from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

# %%
annotations = load_dataset(
    "csv",
    data_files=os.path.join("data", "union_dataset.csv")
).remove_columns(
    "Unnamed: 0"
)

# %%
annotations['train'][0]

# %%
LOCs = annotations.filter(lambda x: x["label"] == "LOC")

# %%
raw_text = load_dataset(
    "csv",
    data_files=os.path.join("data", "text.csv")
).remove_columns(
    "Unnamed: 0"
)

# %%
raw_text['train'][0]

# %%
checkpoint = "dbmdz/bert-base-historic-multilingual-cased"
tokeniser = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModel.from_pretrained(checkpoint)

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

model.to(device)

# %%
tokeniser(raw_text['train'][10]['text'])


# %%
def tokenise_function(row):
    return tokeniser(row['text'])


# %%
tokenised_text = raw_text.map(tokenise_function, batched=True)


# tokenised_text = tokenised_text.remove_columns(["token_type_ids", "attention_mask"])


# %%
def print_tokenised_row(row):
    print(
        row['input_ids'],
        tokeniser.convert_ids_to_tokens(row['input_ids']),
        row['token_type_ids'],
        row['attention_mask'],
        sep='\n'
    )


# %%
print_tokenised_row(tokenised_text['train'][9])
