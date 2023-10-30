#%% Import Packages
import os
from typing import Tuple

import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BatchEncoding

#%% Read data
rawText = (
    pd.read_csv(filepath_or_buffer=os.path.join("data", "text.csv"))
    .drop(columns="Unnamed: 0")
    .set_index(keys="document_id")
)
annotations = (
    pd.read_csv(filepath_or_buffer=os.path.join("data", "union_dataset.csv"))
    .drop(columns="Unnamed: 0")
    .set_index(keys="annotation_id")
)


#%% Get yourself a tokeniser
checkpoint: str = "dbmdz/bert-base-historic-multilingual-cased"
tokeniser: BertTokenizerFast = AutoTokenizer.from_pretrained(checkpoint)
tokeniser.is_fast


#%% Define some helper functions
def print_aligned(words: list, tags: list):
    line1 = ""
    line2 = ""
    for word, tag in zip(words, tags):
        max_length = max(len(word), len(tag))
        line1 += word + " " * (max_length - len(word) + 1)
        line2 += tag + " " * (max_length - len(tag) + 1)
    print(line1)
    print(line2)


#%% prepare label-specific datasets
present_labels = (
    annotations
    .filter(['label'])
    .drop_duplicates()
    .sort_values('label')
    .to_numpy()
)
present_labels = [x[0] for x in present_labels]
for label in present_labels:
    subset = annotations.query(f'label == "{label}"')
    print(label, len(subset))

    # Create list of annotations per line
    subset = (
        subset
        .set_index('line_id')
        .groupby('line_id')[['start', 'end', 'label']]
        .apply(lambda x: x.to_numpy().tolist())
        .reset_index(name='annotations')
    )
    # many thanks to @mozway on https://stackoverflow.com/a/77243869/13044791

    # Merge annotations onto text
    merged = (
        pd.merge(
            left=rawText,
            right=subset,
            how="outer",
            left_on="document_id",
            right_on="line_id"
        )
        .drop(columns="line_id")
    )

    # Initialise non-annotated lines' annotations columns with empty dummy-list
    merged['annotations'] = (
        merged['annotations']
        .apply(lambda entry: entry if isinstance(entry, list) else list())
    )
    # https://stackoverflow.com/a/43899698/13044791

    """
    Desired structure is a table of the following structure per row
    id          int         0
    tokens      sequence    ["EU", "rejects", "German", "call", ...]
    ner_tags    sequence    [   3,         0,        7,      0, ...]
    """

    for index, row in merged.iterrows():
        text: str = row['text']
        annotations = row['annotations']

        inputs: BatchEncoding = tokeniser(text)
        if len(annotations) >= 1:
            x = map_annotations(inputs, annotations)






# %%



# %%
print("Finished")