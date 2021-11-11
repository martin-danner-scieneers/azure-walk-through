"""
Created in November 2021

Python code for finetuning a pretrained bert model

@author: Martin Danner
@company: scieneers GmbH
@mail: martin.danner@scieneers.de
"""


from azureml.core.model import Model
from azureml.core import Dataset, Datastore, Workspace
from azureml.data.datapath import DataPath
from azureml.core import Dataset, Datastore, Run, Workspace
import logging
import argparse
from azureml.core.run import _OfflineRun
from typing import Tuple


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


def get_model_file_path(datastore: Datastore, path: str):
    """Gets the model file from the datastore and returns a pytorch model 

    Args:
        datastore (Datastore): Current datastore to work with
        path (str): Path to the desired model file

    Returns:
       model_file_path: consumable file path of the specified model
    """    
    input_path = DataPath(datastore=datastore, path_on_datastore=path)
    ds = Dataset.File.from_files(path=input_path)
    model_file_path = ds.download()[0]
    
    return model_file_path


def register_model(ws:Workspace, model_name:str, model_path, tags_dict: dict) -> None:
    logging.info('Model registration started!')
    model = Model.register(ws, model_path, model_name, tags_dict)
    logging.info('Model registration done!')


def main():
    parser = argparse.ArgumentParser(description="Process input arguments.")
    parser.add_argument("-mn", "--model_name", type=str, help="Name of the model including filename extension .h5.")
    parser.add_argument("-mfpob", "--model_file_path_on_blob", type=str, help="Path to the model on datastore.")
    parser.add_argument("-t", "--tag", type=str, help="Additional tag to be added to the registered model" )
    parser.add_argument("-dsn", "--datastore_name", type=str, help="Name of the datastore where the model file can be found" )


    args = parser.parse_args()

    model_name = args.model_name
    model_file_path_on_blob = args.model_file_path_on_blob
    tag = args.tag
    datastore_name = args.datastore_name



    # Get the default workspace
    ws, _ = get_current_workspace()
    print(f"Current workspace: {ws}")

    # Get the default datastore
    datastore = Datastore.get(ws, datastore_name=datastore_name)
    print(f"Using Datastore: {datastore.name}")

    # Get the datastore file path in usable format for the registration
    model_file_path = get_model_file_path(datastore, model_file_path_on_blob)

    # Add a custom tag to the registrated model 
    tags_dict = {'Custom Tag': tag}

    # Register the model
    register_model(ws, model_name, model_file_path, tags_dict)

if __name__ == '__main__':
    main()

