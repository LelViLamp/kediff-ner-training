# %% Import Packages
import os
import pandas as pd

# %% Read text CSV
rawText = pd.read_csv(
    filepath_or_buffer=os.path.join("data", "text.csv")
).drop(
    columns="Unnamed: 0"
).set_index(
    keys="document_id"
)

# %% Read annotations CSV
annotations = pd.read_csv(
    filepath_or_buffer=os.path.join("data", "union_dataset.csv")
).drop(
    columns="Unnamed: 0"
).set_index(
    keys="annotation_id"
)

# %%
squash_cols = ['start', 'end', 'label']
annotations = (
    annotations
    .set_index('line_id')
    .groupby('line_id')[squash_cols]
    .apply(
        lambda x: x.to_numpy().tolist()
    )
    .reset_index(name='annotation_list')
)
# many thanks to @mozway on https://stackoverflow.com/a/77243869/13044791
