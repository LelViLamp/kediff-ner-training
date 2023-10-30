#%% Import Packages
import os
from typing import Tuple

import pandas as pd
from transformers import AutoTokenizer, AutoModel, BertTokenizerFast, BatchEncoding

#%% Read data
raw_text_df = (
    pd.read_csv(filepath_or_buffer=os.path.join("data", "text.csv"))
    .drop(columns="Unnamed: 0")
    .set_index(keys="document_id")
)
annotations_df = (
    pd.read_csv(filepath_or_buffer=os.path.join("data", "union_dataset.csv"))
    .drop(columns="Unnamed: 0")
    .set_index(keys="annotation_id")
)


#%% Get yourself a tokeniser
checkpoint: str = "dbmdz/bert-base-historic-multilingual-cased"
tokeniser: BertTokenizerFast = AutoTokenizer.from_pretrained(checkpoint)
print(tokeniser.is_fast)


#%% Define some helper functions
def print_aligned(list1: list, list2: list):
    line1 = ""
    line2 = ""
    for item1, item2 in zip(list1, list2):
        max_length = max(len(item1), len(item2))
        line1 += item1 + " " * (max_length - len(item1) + 1)
        line2 += item2 + " " * (max_length - len(item2) + 1)
    print(line1)
    print(line2)


#%% prepare label-specific datasets
# which labels are there in the dataset?
present_labels = (
    annotations_df
    .filter(['label'])
    .drop_duplicates()
    .sort_values('label')
    .to_numpy()
)
# unpack them
present_labels = [x[0] for x in present_labels]
for label in present_labels:
    subset = annotations_df.query(f'label == "{label}"')
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
            left=raw_text_df,
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

    for _, row in merged.iterrows():
        text: str = row['text']
        annotations = row['annotations']

        annotation_word_spans = []

        inputs: BatchEncoding = tokeniser(text)
        if len(annotations) >= 1:
            # assign annotations to words
            for annotation in annotations:
                # find start and end word
                start_char_annotation = annotation[0]
                end_char_annotation = annotation[1]

                start_word_index = None
                end_word_index = None
                for index, span in enumerate(inputs.encodings[0].offsets):
                    start_char_span: int = span[0]
                    end_char_span: int = span[1]

                    if start_word_index is None:
                        if start_char_annotation <= start_char_span:
                            start_word_index = inputs.encodings[0].word_ids[index]
                    if end_word_index is None:
                        if end_char_annotation == end_char_span:
                            end_word_index = inputs.encodings[0].word_ids[index]
                        elif end_char_annotation < end_char_span:
                            end_word_index = inputs.encodings[0].word_ids[index - 1]
                    if start_word_index is not None and end_word_index is not None:
                        word_span: Tuple[int, int] = (start_word_index, end_word_index)
                        annotation_word_spans.append(word_span)
                        annotation.append(word_span)

                        # check
                        print(text[start_char_annotation:end_char_annotation])
                        # this next line is not an encoded word but just the word index in inputs/encoding/tokenised
                        print(inputs.token_to_word(word_id) for word_id in range(start_word_index, end_word_index))
                        break
                    # find end word
                    # store that word id + tag
                    pass






# %%



# %%
print("Finished")