# %% Import Packages
import os
import pandas as pd

# %% Read text CSV
rawText = (
    pd.read_csv(filepath_or_buffer=os.path.join("data", "text.csv"))
    .drop(columns="Unnamed: 0")
    .set_index(keys="document_id")
)

# %% Read annotations CSV
annotations = (
    pd.read_csv(filepath_or_buffer=os.path.join("data", "union_dataset.csv"))
    .drop(columns="Unnamed: 0")
    .set_index(keys="annotation_id")
)

# %% Create list of annotations per line
annotations = (
    annotations
    .set_index('line_id')
    .groupby('line_id')[['start', 'end', 'label']]
    .apply(lambda x: x.to_numpy().tolist())
    .reset_index(name='annotations')
)
# many thanks to @mozway on https://stackoverflow.com/a/77243869/13044791

# %% Merge annotations onto text
merged = (
    pd.merge(
        left=rawText,
        right=annotations,
        how="outer",
        left_on="document_id",
        right_on="line_id"
    )
    .drop(columns="line_id")
)

# %% Initialise non-annotated lines' annotations columns with empty dummy-list
merged['annotations'] = (
    merged['annotations']
    .apply(lambda x: x if isinstance(x, list) else [])
)
# https://stackoverflow.com/a/43899698/13044791

# %% 
