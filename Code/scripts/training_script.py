"""
Created in November 2021

Python code for finetuning a pretrained bert model

@author: Martin Danner
@company: scieneers GmbH
@mail: martin.danner@scieneers.de
"""

from azureml.data import OutputFileDatasetConfig
from transformers import  AutoModelForSequenceClassification, TrainingArguments, Trainer
from azureml.core import Run, Workspace, Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import logging
import argparse
from azureml.core.run import _OfflineRun
from typing import Tuple
import os
import ntpath 
import torch
import pickle

logging.getLogger().setLevel(logging.DEBUG)


def get_current_workspace() -> Tuple[Workspace, Run]:
    """Gets the current AzureML workspace

    Returns:
        Tuple[Workspace, Run]: [Default Azure Workspace and Run representing the current trial of an the underlying experiment]
    """      
    run = Run.get_context()
    if type(run) == _OfflineRun:
        from Code.utils.env_variables import Env
        e = Env()
        ws = Workspace.get(
            name=e.ws_name,
            subscription_id=e.ws_subscription_id,
            resource_group=e.ws_resource_group,
        )
    else:
        ws = run.experiment.workspace
    return ws, run

def get_training_dataset_by_name(ws: Workspace, dataset_name: str):
    
    ds = Dataset.get_by_name(ws, name=dataset_name)
    ds_path = ds.download()[0]
    with open(ds_path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def write_model(model, model_name: str, run: Run, output_config: OutputFileDatasetConfig, output: str):
    """Outputs the processed data to the defined output (pipeline or offline run)

    Args:
        model (torch.model): model to be stored to the given output
        model_name (str): Name of the model
        run (Run): Run of the current experiment
        output_config (OutputFileDatasetConfig): Class which represents how to copy the output of a run and be promoted as a FileDataset
        output (str): Path where to write the output
    """    
    if type(run) == _OfflineRun:
        _, filename = ntpath.split(output)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        torch.save(model, os.path.join(output, model_name))
    else:
        os.makedirs(os.path.dirname(os.path.join(output_config, output)), exist_ok=True)
        torch.save(model, os.path.join(output_config, output, model_name))


def train_bert_model():

    logging.info('Preprocessing training data!')

    dataset = load_dataset("imdb")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)


    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(16))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(16))

    logging.info('Data ready for training. Finetuning of pretrained bert model started!')

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)

    training_args = TrainingArguments("test_trainer")

    trainer = Trainer(model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset)

    trainer.train()

    logging.info("Model training succesful!")

    return model

def main():
    parser = argparse.ArgumentParser(description="Process input arguments.")
    parser.add_argument("-mn", "--model_name", type=str, help="Name of the model to be trained")
    parser.add_argument("-dn", "--dataset_name", type=str, help="Name of the registered dataset") 
    parser.add_argument("-mpob", "--model_path_on_blob", type=str, help="Path to the model on datastore.")
    parser.add_argument("-o", "--output", type=str)

    args = parser.parse_args()

    model_name = args.model_name
    dataset_name = args.dataset_name
    model_path_on_blob = args.model_path_on_blob
    output = args.output

    # Get the default workspace
    ws, run = get_current_workspace()
    print(f"Current workspace: {ws}")

    #dataset = get_training_dataset_by_name(ws, dataset_name)

    model = train_bert_model()
    write_model(model, model_name, run, output, model_path_on_blob)

if __name__ == '__main__':
    main()
