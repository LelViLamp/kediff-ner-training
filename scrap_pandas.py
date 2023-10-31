# %% duty-free imports
import os
from collections import Counter

import pandas as pd
import plotly.express as px
from tokenizers import Encoding
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizerFast

# %% read data from CSVs
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

# %% get yourself a tokeniser
checkpoint: str = "dbmdz/bert-base-historic-multilingual-cased"
tokeniser: BertTokenizerFast = AutoTokenizer.from_pretrained(checkpoint)
print(tokeniser.is_fast)

# %% which labels are there in the dataset?
present_labels = (
    annotations_df
    .filter(['label'])
    .drop_duplicates()
    .sort_values('label')
    .to_numpy()
)
# unpack them, cause rn this is a list of one-element lists
present_labels = [x[0] for x in present_labels]


# %% define some helper functions
def print_aligned(
        list1: list,
        list2: list
):
    line1 = ""
    line2 = ""
    for item1, item2 in zip(list1, list2):
        max_length = max(len(item1), len(item2))
        line1 += item1 + " " * (max_length - len(item1) + 1)
        line2 += item2 + " " * (max_length - len(item2) + 1)
    print(line1)
    print(line2)


def get_subset_data(
        annotations_df,
        label: str
):
    subset = annotations_df.query(f'label == "{label}"')
    # Create list of annotations per line
    subset = (
        subset
        .set_index('line_id')
        .groupby('line_id')[['start', 'end', 'label']]
        .apply(lambda x: x.to_numpy().tolist())
        .reset_index(name='annotations')
    )
    # many thanks to @mozway on https://stackoverflow.com/a/77243869/13044791
    return subset


def merge_annotations_and_text(
        raw_text_df,
        annotations_df
):
    # Merge annotations onto text
    merged = (
        pd.merge(
            left=raw_text_df,
            right=annotations_df,
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
    return merged


def annotations_to_token_BILUs(
        tokenised: Encoding,
        annotations
):
    # https://www.lighttag.io/blog/sequence-labeling-with-transformers/example
    tokens = tokenised.tokens
    # make a list to store our labels the same length as our tokens
    aligned_labels = ["O"] * len(tokens)

    for annotation in annotations:
        start = annotation[0]
        end = annotation[1]
        label = annotation[2]

        # a set that stores the token indices of the annotation
        annotation_token_index_set = (set())
        for char_index in range(start, end):
            token_index = tokenised.char_to_token(char_index)
            if token_index is not None:
                annotation_token_index_set.add(token_index)
        if len(annotation_token_index_set) == 1:
            # if there is only one token
            token_index = annotation_token_index_set.pop()
            prefix = ("U")  # This annotation spans one token so is prefixed with U for unique
            aligned_labels[token_index] = f"{prefix}-{label}"
        else:
            last_token_in_anno_index = len(annotation_token_index_set) - 1
            for num, token_index in enumerate(sorted(annotation_token_index_set)):
                if num == 0:
                    prefix = "B"
                elif num == last_token_in_anno_index:
                    prefix = "L"  # Its the last token
                else:
                    prefix = "I"  # We're inside of a multi token annotation
                aligned_labels[token_index] = f"{prefix}-{label}"
    return aligned_labels


# %% tokenise
merged_dict = {'text': raw_text_df['text'].values.tolist()}
token_counter = Counter()
tokenised = []
for text in tqdm(merged_dict['text'], desc="Tokenise"):
    tokenised_text = tokeniser(text)[0]
    tokens = tokenised_text.tokens

    tokenised.append(tokenised_text)
    token_counter.update(tokens)
merged_dict['tokenised'] = tokenised

# %% plot token_counter and appreciate its Zipf-iness
token_counter_df = pd.DataFrame.from_dict([token_counter]).transpose()
px.histogram(token_counter_df).show()
(
    px
    .bar(token_counter_df, x=token_counter_df.index, y=0)
    .update_layout(xaxis={'categoryorder': 'total descending'})
    .show()
)
# %% annotation columns per label
longest_label_length: int = len(max(present_labels, key=len))
for label in present_labels:
    label_subset_df: pd.DataFrame = get_subset_data(annotations_df, label)
    subset_BILUs = len(merged_dict['text']) * [list()]

    for _, row in tqdm(label_subset_df.iterrows(), total=len(label_subset_df),
                       desc="Align " + str.ljust(label, longest_label_length)):
        line_id: int = row['line_id']
        dict_access_index: int = line_id - 1
        annotations = row['annotations']

        tokenised_text = merged_dict['tokenised'][dict_access_index]
        BILUs = annotations_to_token_BILUs(tokenised_text, annotations)
        subset_BILUs[dict_access_index].append(BILUs)

    merged_dict[f'{label}-BILUs'] = subset_BILUs

# %%
print("Finished")
