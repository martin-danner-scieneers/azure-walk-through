"""
Created in November 2021

Python code for finetuning a pretrained bert model

@author: Martin Danner
@company: scieneers GmbH
@mail: martin.danner@scieneers.de
"""

from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.model import Model
from azureml.core import Dataset, Datastore, Workspace
from azureml.data.datapath import DataPath
from azureml.data import OutputFileDatasetConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from azureml.core import Dataset, Datastore, Run, Workspace
import numpy as np
from datasets import load_metric
import torch.onnx
import logging
import argparse
from azureml.core.run import _OfflineRun
from typing import Tuple
import os
import ntpath 

logging.getLogger().setLevel(logging.DEBUG)


def get_current_workspace() -> Tuple[Workspace, Run]:
    """Gets the current AzureML workspace

    Returns:
        Tuple[Workspace, Run]: [Default Azure Workspace and Run representing the current trial of an the underlying experiment]
    """      
    run = Run.get_context()
    if type(run) == _OfflineRun:
        from utils.env_variables import Env
        e = Env()
        #interactive_auth  = InteractiveLoginAuthentication(tenant_id=e.tenant_id)
        ws = Workspace.get(
            name=e.ws_name,
            subscription_id=e.ws_subscription_id,
            resource_group=e.ws_resource_group,
            #auth=interactive_auth
        )
    else:
        ws = run.experiment.workspace
    return ws, run

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
    logging.info('Retrieving training data!')

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

    trainer = Trainer(model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset = small_eval_dataset)

    trainer.train()

    logging.info("Model training succesful!")

    return model

def main():
    parser = argparse.ArgumentParser(description="Process input arguments to interact with the underlying Seq2Seq architecture.")
    parser.add_argument("-mn", "--model_name", type=str, help="Name of the model including filename extension .h5.")
    parser.add_argument("-mpob", "--model_path_on_blob", type=str, help="Path to the model on datastore.")
    parser.add_argument("-o", "--output", type=str)

    args = parser.parse_args()

    model_name = args.model_name
    model_path_on_blob = args.model_path_on_blob
    output = args.output


    # Get the default workspace
    ws, run = get_current_workspace()
    print(f"Current workspace: {ws}")

    # Get the default datastore
    datastore = Datastore.get_default(ws)
    print(f"Using Datastore: {datastore.name}")

    model = train_bert_model()
    write_model(model, model_name, run, output, model_path_on_blob)
    


if __name__ == '__main__':
    main()

