# preferrably imported as kners as it sounds fun
# generated from step3_production_pipeline.ipynb

import os
from typing import Any

import tabulate
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline, Pipeline

TOKENISER_CHECKPOINT = "dbmdz/bert-base-historic-multilingual-cased"
CLASSIFIER_NAME_BASE = "oalz-1788-q1-ner-"
CLASSIFIER_MODEL_VERSION = "2024-01-15"
ENTITY_TYPES = ["EVENT", "LOC", "MISC", "ORG", "PER", "TIME"]
SELECTED_EPOCHS = {
    "EVENT": "checkpoint-1393",
    "LOC": "checkpoint-1393",
    "MISC": "checkpoint-2786",
    "ORG": "checkpoint-1393",
    "PER": "checkpoint-2786",
    "TIME": "checkpoint-1393"
}


class KediffNerSystem:
    """
    Wrap several NER classifiers to be run on the same text to combine the result. Thus, allows multi-label
    classification by using an ensemble of single-class classifiert, the results of which are combined by unioning the
    individual results without further checks.

    The main method of this class that you want to use after initialisation is ner.
    """

    tokeniser_checkpoint: str
    tokeniser: AutoTokenizer
    print_debug_messages_to_console: bool
    classifier_paths: dict[str, str]
    classifiers: dict[str, Pipeline]

    def __init__(self,
                 classifier_paths: dict[str, str],
                 print_debug_messages_to_console: bool = False,
                 tokeniser_checkpoint: str = "dbmdz/bert-base-historic-multilingual-cased"):
        """
        Initialise the NER System built on the KEDiff OALZ 1788/Q1 NER dataset. Can be used
        for other datasets as well.

        :param classifier_paths:
            Provide paths to the trained models/classifiers. Should be a dictionary of entity
            type and path to the directory of the trained model/epoch of that model.
        print_debug_messages_to_console:
            Should messages be printed to the console while initialising the class? Can be a first
            step in debugging, can also be annoying to the end user.
        :param tokeniser_checkpoint:
            Should not be changed as the models specified in classifier_paths probably expect this
            specific tokeniser. Changing it might render your output gibberish.
        """

        self.print_debug_messages_to_console = print_debug_messages_to_console

        self.tokeniser_checkpoint = tokeniser_checkpoint
        if self.print_debug_messages_to_console:
            print(f"Loading tokeniser '{self.tokeniser_checkpoint}'")
        self.tokeniser = AutoTokenizer.from_pretrained(tokeniser_checkpoint)

        # initialise the models
        if self.print_debug_messages_to_console:
            print(f"Initialising models. Received paths to {len(classifier_paths)} classifiers")
        classifiers: dict[str, Pipeline] = {}
        for label_type in tqdm(classifier_paths.keys(), disable=not self.print_debug_messages_to_console):
            classifiers[label_type] = pipeline(
                "token-classification",
                model=os.path.abspath(classifier_paths[label_type]),
                aggregation_strategy="simple"
            )
        self.classifier_paths = classifier_paths
        self.classifiers = classifiers

        if self.print_debug_messages_to_console:
            print(f"Class initialised")

    def __call__(self, *args: Any, **kwds: Any) -> list[Any]:
        return self.ner(text=args[0])

    def find_entities(self, text: str) -> dict[str, Any]:
        """
        Find the entities in the text using the models specified and initialised when initialising the class.
        The resulting dictioanry's entries have the following keys and data types:

        * entity_type: str
        * score: double
        * word: str
        * start: int
        * end: int

        :param text: The text in which to find entities.
        :return: Dictionary per entity type and the found entities.
        """

        entities: dict[str, Any] = {
            label_type: self.classifiers[label_type](text)
            for label_type in ENTITY_TYPES
        }
        return entities

    def entities_dict_to_list(self, entities_dict: dict[str, Any]) -> list[Any]:
        """
        Convert the entity dictionary to a list, i.e. remove the distiction by the dictionary's keys.

        :param entities_dict: Dictionary of which the keys are to be dropped
        :return: All the entries/values of the dictioanry appended as a list.
        """
        entities_list: list[Any] = []
        for label_type in entities_dict:
            entities_list += entities_dict[label_type]
        return entities_list

    def sort_entity_list(self, entity_list: list[Any]) -> list[Any]:
        """
        Sort a list of entities by its start index, score and entity_group.

        :param entity_list: List of entities.
        :return: A new list sorted as specified above
        """

        sorted_entity_list: list[Any] = sorted(entity_list, key=lambda d: (d["start"], d["score"], d["entity_group"]))
        return sorted_entity_list

    def print_entities_as_table(self,
                                entity_list: list[Any],
                                text_when_empty: str = "(no entities found)",
                                tablefmt: str = "simple_outline") -> None:
        """
        Prints a list of entities as a table to the console using tabulate. Assumes all entries in the list have the
        same keys in the same order. If the list is empty, an alternative text can be shown.

        :param entity_list:
            The entities to be printed as a table to the console.
            All entries in this list should have the same format.
        :param text_when_empty:
            Alternative text in case entity_list is empty. Can be set to an empty string to skip printing.
        :param tablefmt: see tabluate.tabulate's parameter documentation.
        :return: Nothing, only prints to the console.
        """

        if entity_list is None or len(entity_list) == 0:
            if type(text_when_empty) is str and len(text_when_empty) >= 1:
                print(text_when_empty)
            return
        header = list(entity_list[0].keys())
        header[0] = "type"
        rows = [entity.values() for entity in entity_list]
        print(tabulate.tabulate(rows, header, tablefmt=tablefmt))

    def ner(self,
            text: str,
            print_table_to_console: bool = False) -> list[Any]:
        """
        Find named entities in the given text using this wrapper's models, aggregate and sort found entities.

        :param text: The text that is to be run through all the classifiers.
        :param print_table_to_console: Should the result be printed to the console as a table?
        :return: Named entities found in the text provided.
        """

        entity_dict: dict[str, Any] = self.find_entities(text)
        entity_list: list[Any] = self.entities_dict_to_list(entity_dict)
        sorted_entity_list: list[Any] = self.sort_entity_list(entity_list)
        if print_table_to_console:
            self.print_entities_as_table(sorted_entity_list)
        return sorted_entity_list


if __name__ == "__main__":
    DATA_DIR = os.path.join('data')
    TRAINED_DIR = os.path.join(DATA_DIR, "trained_models", CLASSIFIER_MODEL_VERSION)

    ner_model_paths = {
        entity_type: os.path.join(TRAINED_DIR,
                                  "".join([CLASSIFIER_NAME_BASE, entity_type]),
                                  SELECTED_EPOCHS[entity_type])
        for entity_type in ENTITY_TYPES
    }
    sample_texts = [
        "(Das hei\u00dft ab ovo anfangen, wie's jener that, der vom deutschen Gleichgewichte handeln wollte, "
        "und von Adam anfieng.)",

        "Daniel Göller ist der beste Masterstudent Christian Borgelts und sollte eine Ehrenmedaille… medallje… "
        "medallie… wie schreibt man das????… sowie eine saftige Sonderzahlung von Herrn Prof. Lehnert, Rektor der "
        "Universität zu Salzburg, erhalten, sodass er endlich nach Island reisen und dort in einer Kirche zu Gott "
        "beten kann.",

        "Nun ist die Frage, ob das Modell auch mit Frauennamen umgehen kann, da beim Lesen der Originaltexte ein "
        "deutlicher Bias zu Männernamen aufgefallen ist. Und wie es dann wohl mit geschlechtsneutralen Namen aussieht?",

        "Bundeskanzlerin Brigitte Bierlein führte bis zur Angelobung der Bundesregierung Kurz II nach der "
        "vorgezogenen Nationalratswahl im Herbst 2019 die Amtsgeschäfte der Bundesministerien weiter. Vielleicht "
        "kennt sie ja auch Angela Merkel?",

        "Test in Salzburg während der Österreichsichen Aufklärung. In Paris wurden mehrere Menschen aus Deutschland "
        "gesichtet.",

        "den meisten Lesern durch eine ausführliche Beschreibung und Beurtheilung des Wirtembergischen, "
        "im katholischen Deutschlande noch immer nicht genug belannten Gesangbuches, einen Gefallen zu erzeigen."
    ]

    kediff_ner = KediffNerSystem(ner_model_paths, print_debug_messages_to_console=True)

    for i, t in enumerate(sample_texts):
        print(t)
        kediff_ner.ner(t, print_table_to_console=True)
