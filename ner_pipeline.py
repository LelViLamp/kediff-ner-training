from transformers import AutoModelForTokenClassification, pipeline

# initialise models, which include the tokeniser
label_types = ["EVENT", "LOC", "MISC", "ORG", "PER", "TIME"]
classifier = [
    AutoModelForTokenClassification.from_pretrained(f"LelViLamp/OALZ-1788-Q1-NER-{label_type}")
    for label_type in label_types
]

# initialise pipeline
pipeline = [
    pipeline(
        task="token-classification",
        model=classifier[label_type],
        aggregation_strategy="simple"
    )
    for label_type in label_types
]


def find_entities(text, pipeline):
    return {
        label_type: pipeline[label_type](text)
        for label_type in pipeline.keys()
    }
