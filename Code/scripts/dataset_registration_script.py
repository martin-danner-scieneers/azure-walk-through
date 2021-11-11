"""
Created in November 2021

Python code to register datasets from datastore

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

def get_dataset_file(datastore: Datastore, path: str):
    """Gets the dataset file from the datastore and returns a dataset

    Args:
        datastore (Datastore): Current datastore to work with
        path (str): Path to the desired model file

    Returns:
       model_file_path: consumable file path of the specified model
    """    
    input_path = DataPath(datastore=datastore, path_on_datastore=path)
    ds = Dataset.File.from_files(path=input_path)
    
    return ds

def register_dataset(datastore: Datastore, ws: Workspace, dataset_path_on_blob: str, dataset_name:str) -> None:
    logging.info('Dataset registration started!')
    dataset = get_dataset_file(datastore, dataset_path_on_blob)
    dataset.register(ws, name=dataset_name)
    logging.info('Dataset registration done!')


def main():
    parser = argparse.ArgumentParser(description="Process input arguments.")
    parser.add_argument("-dn", "--dataset_name", type=str, help="Name of the dataset to be registered")
    parser.add_argument("-dspob", "--dataset_path_on_blob", type=str, help="Path to the dataset on datastore.")
    parser.add_argument("-dsn", "--datastore_name", type=str, help="Name of the datastore where the model file can be found " )


    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_path_on_blob = args.dataset_path_on_blob
    datastore_name = args.datastore_name



    # Get the default workspace
    ws, _ = get_current_workspace()
    print(f"Current workspace: {ws}")

    # Get the default datastore
    datastore = Datastore.get(ws, datastore_name=datastore_name)
    print(f"Using Datastore: {datastore.name}")

    # Register dataset in AzureML
    register_dataset(datastore, ws, dataset_path_on_blob, dataset_name)


if __name__ == '__main__':
    main()

