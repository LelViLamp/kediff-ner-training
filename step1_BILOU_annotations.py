# %% duty-free imports
import os
from collections import Counter
from typing import List, Any, Union, Set, Dict

import datasets
import pandas as pd
from datasets import Dataset, ClassLabel, Features
from numpy import ndarray
from pandas import DataFrame
from tokenizers import Encoding
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizerFast
from transformers.tokenization_utils_base import EncodingFast

# %% some parameters
model_checkpoint: str = "dbmdz/bert-base-historic-multilingual-cased"

# %% read data from CSVs
raw_text_df: DataFrame = (
    pd.read_csv(filepath_or_buffer=os.path.join("data", "text.csv"))
    .drop(columns="Unnamed: 0")
    .set_index(keys="document_id")
)
annotations_df: DataFrame = (
    pd.read_csv(filepath_or_buffer=os.path.join("data", "union_dataset.csv"))
    .drop(columns="Unnamed: 0")
    .set_index(keys="annotation_id")
)

# %% get yourself a tokeniser
tokeniser: BertTokenizerFast = AutoTokenizer.from_pretrained(model_checkpoint)
print("This is a fast tokeniser?", tokeniser.is_fast)

# %% which labels are there in the dataset?
present_labels_df: ndarray = (
    annotations_df
    .filter(['label'])
    .drop_duplicates()
    .sort_values('label')
    .to_numpy()
)
# unpack them, cause rn this is a list of one-element lists
present_labels: list[str] = [x[0] for x in present_labels_df]


# %% define some helper functions
def get_subset_data(
        annotations_df: DataFrame,
        label: str
) -> DataFrame:
    subset: DataFrame = annotations_df.query(f'label == "{label}"')
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
    # end def


def merge_annotations_and_text(
        raw_text_df: DataFrame,
        annotations_df: DataFrame
) -> DataFrame:
    # Merge annotations onto text
    merged: DataFrame = (
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
    # end def


def annotations_to_token_BILOUs(
        tokenised: Encoding,
        annotations: list[list[list[Union[int, str]]]]
) -> List[str]:
    # https://www.lighttag.io/blog/sequence-labeling-with-transformers/example
    tokens: list[str] = tokenised.tokens
    # make a list to store our labels the same length as our tokens
    aligned_labels: list[str] = ["O"] * len(tokens)

    if len(annotations) == 0:
        return aligned_labels
    else:
        annotation_subset: list[list[Union[int, str]]] = annotations[0]

    annotation: list[Union[int, str]]
    for annotation in annotation_subset:
        start: int = annotation[0]
        end: int = annotation[1]
        label: str = annotation[2]

        # a set that stores the token indices of the annotation
        annotation_token_index_set: set[int] = (set())
        char_index: int
        for char_index in range(start, end):
            token_index: int = tokenised.char_to_token(char_index)
            if token_index is not None:
                annotation_token_index_set.add(token_index)
            # end for
        if len(annotation_token_index_set) == 1:
            # if there is only one token
            token_index: int = annotation_token_index_set.pop()
            prefix: str = ("U")  # This annotation spans one token so is prefixed with U for unique
            aligned_labels[token_index] = f"{prefix}-{label}"
        else:
            last_token_in_anno_index: int = len(annotation_token_index_set) - 1
            num: int
            for num, token_index in enumerate(sorted(annotation_token_index_set)):
                prefix: str
                if num == 0:
                    prefix = "B"  # beginning
                elif num == last_token_in_anno_index:
                    prefix = "L"  # it's the last token
                else:
                    prefix = "I"  # We're inside a multi token annotation
                aligned_labels[token_index] = f"{prefix}-{label}"
                # end for
            # end else
        # end for
    return aligned_labels
    # end def


def convert_BILOUs_to_IOBs(BILOUs: List[str]) -> List[str]:
    """
    Converts BILOU annotations into IOB annotations by replacing "L-" tags with "I-" and "U-" tags with "B-"

    :param BILOUs: List of BILOU tags as strings
    :return: List of IOB tags, as long as input list
    """
    IOBs: List[str] = []

    tag: str
    for tag in BILOUs:
        if tag.startswith("L"):
            tag = "I" + tag.removeprefix("L")
        elif tag.startswith("U"):
            tag = "B" + tag.removeprefix("U")
        IOBs.append(tag)
        # end for
    return IOBs
    # end def


# %% tokenise
dataset_dict: dict[str, Union[Encoding, str, list[str]]] = {'Text': raw_text_df['text'].values.tolist()}
token_counter: Counter[str] = Counter()
tokenised: list[Encoding] = []
text: str
for text in tqdm(dataset_dict['Text'], desc="Tokenise all texts"):
    tokenised_text: Encoding = tokeniser(text)[0]
    tokens: list[str] = tokenised_text.tokens

    tokenised.append(tokenised_text)
    token_counter.update(tokens)
    # end for
dataset_dict['tokenised'] = tokenised

# %% store token_counter statistics, plot them somewhere else to appreciate its Zipf-iness
token_counter_df: DataFrame = pd.DataFrame.from_dict([token_counter]).transpose()
token_counter_df.to_csv(
    os.path.join('data', 'token_counter.csv'),
    index_label="Token", header=["Count"]
)

# %% BILOU and IOB annotation columns per label
label_to_int: dict[str, int] = {'O': 0}
label: str
for label in present_labels:
    prefix: str
    for prefix in ['B', 'I', 'L', 'U']:
        label_to_int[f"{prefix}-{label}"] = len(label_to_int)
        # end for

    label_subset_df: DataFrame = get_subset_data(annotations_df, label)
    subset_BILOUs: list[list[str]] = [] # todo
    subset_IOBs: list[list[str]] = [] # todo

    dict_access_index: int
    for dict_access_index in tqdm(
            range(len(dataset_dict['Text'])),
            desc=f"Converting {label} annotations to BILOUs and IOBs"
    ):
        line_id: int = dict_access_index + 1
        text: str = dataset_dict['Text'][dict_access_index]
        tokenised_text = dataset_dict['tokenised'][dict_access_index]
        annotations: DataFrame = label_subset_df.query(f"line_id == {line_id}")
        annotations: list[list[int, str]] = annotations['annotations'].values.tolist()

        BILOUs: list[str] = annotations_to_token_BILOUs(tokenised_text, annotations)
        subset_BILOUs.append(BILOUs)

        IOBs: list[str] = convert_BILOUs_to_IOBs(BILOUs)
        subset_IOBs.append(IOBs)
        # end for
    del BILOUs, IOBs

    dataset_dict[f'{label}-BILOUs'] = subset_BILOUs
    dataset_dict[f'{label}-IOBs'] = subset_IOBs
    # end for
del subset_BILOUs, subset_IOBs

# %% export the dataset
if "tokenised" in dataset_dict:
    del dataset_dict['tokenised']
dataset_df: DataFrame = pd.DataFrame.from_dict(dataset_dict)
dataset_df.to_csv(
    os.path.join('data', 'BILOUs.csv'),
    index=False
)

# %% convert to a HuggingFace dataset
ner_class_label: ClassLabel = ClassLabel(
    num_classes=len(label_to_int),
    names=list(label_to_int.keys())
)
features: Features = Features({
    'Text': datasets.Value(dtype='string'),
    'EVENT-BILOUs': datasets.Sequence(feature=ner_class_label, length=-1),
    'EVENT-IOBs': datasets.Sequence(feature=ner_class_label, length=-1),
    'LOC-BILOUs': datasets.Sequence(feature=ner_class_label, length=-1),
    'LOC-IOBs': datasets.Sequence(feature=ner_class_label, length=-1),
    'MISC-BILOUs': datasets.Sequence(feature=ner_class_label, length=-1),
    'MISC-IOBs': datasets.Sequence(feature=ner_class_label, length=-1),
    'ORG-BILOUs': datasets.Sequence(feature=ner_class_label, length=-1),
    'ORG-IOBs': datasets.Sequence(feature=ner_class_label, length=-1),
    'PER-BILOUs': datasets.Sequence(feature=ner_class_label, length=-1),
    'PER-IOBs': datasets.Sequence(feature=ner_class_label, length=-1),
    'TIME-BILOUs': datasets.Sequence(feature=ner_class_label, length=-1),
    'TIME-IOBs': datasets.Sequence(feature=ner_class_label, length=-1)
})

BILOUs_hug: Dataset = Dataset.from_pandas(df=dataset_df, features=features)
print(BILOUs_hug)
print(BILOUs_hug.features)

BILOUs_hug.save_to_disk(dataset_path=os.path.join('data', 'BILOUs_hf'))
