"""
Created in November 2021

Python code for finetuning a pretrained bert model

@author: Martin Danner
@company: scieneers GmbH
@mail: martin.danner@scieneers.de
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.model import Model
from azureml.core import Run, Workspace, Datastore
from azureml.core.run import _OfflineRun
from typing import Tuple
import logging
import argparse
import pickle

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
        interactive_auth  = InteractiveLoginAuthentication(tenant_id=e.tenant_id)
        ws = Workspace.get(
            name=e.ws_name,
            subscription_id=e.ws_subscription_id,
            resource_group=e.ws_resource_group,
            auth=interactive_auth
            )
    else:
        ws = run.experiment.workspace
    return ws, run


def write_runtime(model_name: str, datastore:Datastore, base_dir: str, model_path_on_blob: str):
    """Store model name as runtime param

    Args:
        model_name (str): Name of the model to be deployed
        datastore (str): Datastore to store the runtime information
        model_path_on_blob (str): Path where the model is stored 
    """
    params = {'model_name': model_name}
    with open(os.path.join(base_dir, 'runtime_garage', 'runtime_params.pkl'), 'wb') as file:
        pickle.dump(params, file)

    datastore.upload_files(files=[os.path.join(base_dir, 'runtime_garage', 'runtime_params.pkl')], target_path=model_path_on_blob, overwrite=True, show_progress=False)


def deploy_model(ws: Workspace, model_name:str, service_name:str, entry_script_name:str, base_dir:str, env_dir:str, script_dir:str):
    """Deployment of an already registered model as an azure container instance service

    Args:
        ws (Workspace): [description]
        service_name (str): Name of the azure container instance service
        entry_scipt_name (str): File name of the entry script
        base_dir (str): Base directory containing all necessary files
        env_dir (str): Directory containing the yaml environment file
        script_dir (str): Directory containing the entry script to be executed during inference
    """
    
    # Set environment
    env = Environment.get(workspace=ws, name="AzureML-Minimal")

    print(os.getcwd())
    print(__file__)

    # Create Environment from yaml file
    env_filepath = os.path.join(env_dir, "cl_env.yml")
    env = Environment.from_conda_specification(
        name="Cluster_Env", file_path=env_filepath)

    # Set and specify inference config
    inference_config = InferenceConfig(
        environment=env,
        source_directory=base_dir,
        entry_script=os.path.join(script_dir, entry_script_name)
    )

    # Set and specify deployment config
    try:
        registered_model = Model(ws, model_name)
    except:
        logging.info("There is no registered model with specified name available. Please register the specified model first!")

    deployment_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
    service = registered_model.deploy(ws, service_name, [registered_model], inference_config, deployment_config)
    service.wait_for_deployment(show_output=True)
    logging.info(service.state)
    logging.info("Deployment done!")


def main():
    parser = argparse.ArgumentParser(description="Process input arguments to interact with the underlying Seq2Seq architecture.")
    parser.add_argument("-mn", "--model_name", type=str, help="Name of the model")
    parser.add_argument("-sn", "--service_name", type=str, help="Name of the deployed service")
    parser.add_argument("-dsn", "--datastore_name", type=str, help="Name of the datastore where the runtime file should be stored to" )
    parser.add_argument("-mpob", "--model_path_on_blob", type=str, help="Path to the model on datastore.")

    args = parser.parse_args()

    model_name = args.model_name
    service_name = args.service_name
    datastore_name = args.datastore_name
    model_path_on_blob = args.model_path_on_blob

    # Directory Setup
    base_dir = os.path.dirname(os.path.dirname(__file__))
    env_dir = os.path.join(base_dir, "env")
    script_dir = os.path.join(base_dir, "scripts")

    # Name of the entry script
    entry_script_name = 'entry_script.py'

    # Get the default workspace
    ws, _ = get_current_workspace()
    print(f"Current workspace: {ws}")

    # Get the default datastore
    datastore = Datastore.get(ws, datastore_name=datastore_name)
    print(f"Using Datastore: {datastore.name}")

    # Write runtime
    write_runtime(model_name, datastore, base_dir, model_path_on_blob)

    # Deploy the specified model
    deploy_model(ws, model_name, service_name, entry_script_name, base_dir, env_dir, script_dir)

if __name__ == '__main__':
    main()

