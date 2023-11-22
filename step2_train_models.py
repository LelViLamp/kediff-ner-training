# %% install some stuff on colab
# !pip install datasets evaluate transformers[sentencepiece] seqeval
# !pip install accelerate

# %% duty-free imports
import os

import evaluate
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import BertTokenizerFast, AutoTokenizer, DataCollatorForTokenClassification, \
    AutoModelForTokenClassification, TrainingArguments, Trainer

# %% some parameters
model_checkpoint: str = "dbmdz/bert-base-historic-multilingual-cased"


# %% define some functions
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


# %% set DATA_DIR depending on whether we're working in Colab
try:
    from google.colab import drive

    print(
        "You work on Colab. Gentle as we are, we will mount Drive for you. "
        "It'd help if you allowed this in the popup that opens."
    )
    drive.mount('/content/drive')
    DATA_DIR = os.path.join('drive', 'MyDrive', 'KEDiff', 'data')
except:
    print("You do not work on Colab")
    DATA_DIR = os.path.join('data')
    pass

print(f"{DATA_DIR=}", '-->', os.path.abspath(DATA_DIR))

# %% import dataset
BILOUs_hug = Dataset.load_from_disk(dataset_path=os.path.join(DATA_DIR, 'BILOUs_hf'))
print("Dataset:", BILOUs_hug, sep='\n')
print("Features:", BILOUs_hug.features, sep='\n')

# %% split dataset --> train, test, validation
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

# %% tokenise
tokeniser: BertTokenizerFast = AutoTokenizer.from_pretrained(model_checkpoint)
print(f"Is '{model_checkpoint}' a fast tokeniser?", tokeniser.is_fast)


def batch_embed(batch):
    # align annotation with added [CLS] and [SEP]
    for column in [
        'EVENT-BILOUs', 'LOC-BILOUs', 'MISC-BILOUs', 'ORG-BILOUs', 'PER-BILOUs', 'TIME-BILOUs',
        'EVENT-IOBs', 'LOC-IOBs', 'MISC-IOBs', 'ORG-IOBs', 'PER-IOBs', 'TIME-IOBs'
    ]:
        all_labels = batch[column]
        new_labels = [[-100, *labels[1:-1], -100] for labels in all_labels]
        batch[column] = new_labels
    return batch


BILOUs_hug = BILOUs_hug.map(batch_embed, batched=True)


def batch_tokenise(batch):
    # tokenise
    tokenised_inputs = tokeniser(batch['Text'], truncation=True)
    tokenised_inputs["labels"] = batch['PER-IOBs']
    return tokenised_inputs


BILOUs_hug_tokenised = BILOUs_hug.map(
    batch_tokenise,
    batched=True,
    remove_columns=BILOUs_hug["train"].column_names
)
print(BILOUs_hug_tokenised)

# %% get a sample
sample = BILOUs_hug_tokenised["train"][1]
print(sample)
del sample

# %% collate it
data_collator = DataCollatorForTokenClassification(tokenizer=tokeniser, padding=True)
batch = data_collator([BILOUs_hug_tokenised["train"][i] for i in range(2)])
print(batch)
print(batch['labels'])

for i in range(2):
    print(BILOUs_hug_tokenised["train"][i]["labels"])
del i

# %% test different metrics
label_names = BILOUs_hug["train"].features["PER-IOBs"].feature.names

batch = {'references': [], 'predictions': []}
for i in [0, 1]:
    labels = BILOUs_hug["train"][i]["PER-IOBs"]
    labels = [label_names[i] for i in labels[1:-1]]
    # fake predictions
    predictions = labels.copy()
    predictions[2] = "B-PER"
    predictions[3] = "I-PER"

    print_aligned(labels, predictions)

    batch['references'] += [labels]
    batch['predictions'] += [predictions]
del i, labels, predictions

# calculate metrics
for metric_name in ["seqeval", "poseval"]:
    print(f"Now evaluating using {metric_name=}")
    metric = evaluate.load(metric_name)
    metric_result = metric.compute(predictions=batch['predictions'], references=batch['references'])
    print(metric_result)
# del batch, metric, metric_name, metric_result

# %% choose poseval as metric
metric = evaluate.load('poseval')

# %%
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    metric_result = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "macro precision": metric_result["macro avg"]["precision"],
        "macro recall": metric_result["macro avg"]["recall"],
        "macro f1": metric_result["macro avg"]["f1-score"],
        "macro support": metric_result["macro avg"]["support"],

        "weighted precision": metric_result["weighted avg"]["precision"],
        "weighted recall": metric_result["weighted avg"]["recall"],
        "weighted f1": metric_result["weighted avg"]["f1-score"],
        "weighted support": metric_result["weighted avg"]["support"],

        "accuracy": metric_result["accuracy"],
    }


# %%
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
model.config.num_labels

# %%
trained_model_name = "oalz-1788-q1-ner-PER"
args = TrainingArguments(
    trained_model_name,
    output_dir = os.path.join(DATA_DIR, trained_model_name),
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=10,
    weight_decay=0.01
)

# todo store checkpoints on drive as well not just final model
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=BILOUs_hug_tokenised["train"],
    eval_dataset=BILOUs_hug_tokenised["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokeniser,
)
trainer.train()
trainer.save_model(os.path.join(DATA_DIR, trained_model_name))

# %%
pass
