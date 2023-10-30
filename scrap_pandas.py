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


# a method for adding labels to the tokenised input
def find_word_ids_for_span(tokenised_input: BatchEncoding, span) -> Tuple[int, int]:
    offsets = tokenised_input.encodings[0].offsets
    word_ids = tokenised_input.word_ids()
    start: int = span[0]
    end: int = span[1]

    start_word_id: int = 0
    end_word_id: int = 0

    while start_word_id == 0:
        for idx, tuple in enumerate(offsets):
            ostart = tuple[0]
            if start <= ostart:
                start_word_id = word_ids[idx]
                break
    while end_word_id == 0:
        for idx, tuple in enumerate(offsets):
            oend = tuple[1]
            if end <= oend:
                end_word_id = word_ids[idx]
                break

    return (start_word_id, end_word_id)


def map_annotations(tokenised_input: BatchEncoding, annotations):
    spans = []
    for annotation in annotations:
        word_span = find_word_ids_for_span(tokenised_input, annotation)
        spans.append(word_span)
    
    return spans


def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


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