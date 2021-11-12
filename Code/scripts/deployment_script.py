"""
Created in November 2021

Python code to deploy a model as an Azure Container Instance Webservice

@author: Martin Danner
@company: scieneers GmbH
@mail: martin.danner@scieneers.de
"""
import sys
import os
from azureml.core import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.model import Model
from azureml.core import Run, Workspace
from azureml.core.run import _OfflineRun
from typing import Tuple
import logging
import argparse
from pathlib import Path




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


def deploy_model(ws: Workspace, model_name:str, service_name:str, entry_script_name:str, env_dir:str):
    """Deployment of an already registered model as an azure container instance service

    Args:
        ws (Workspace): Default Workspace
        service_name (str): Name of the azure container instance service
        entry_scipt_name (str): File name of the entry script
        env_dir (str): Directory containing the yaml environment file
    """
    
    # Set environment
    env = Environment.get(workspace=ws, name="AzureML-Minimal")

    print(os.getcwd())
    print(__file__)

    # Create Environment from yaml file
    env_filepath = os.path.join(env_dir, "cl_env.yml")
    logging.info(env_filepath)
    env = Environment.from_conda_specification(
        name="Cluster_Env", file_path=env_filepath)

    # Set and specify inference config
    inference_config = InferenceConfig(
        environment=env,
        entry_script=entry_script_name)
    

    # Set and specify deployment config
    try:
        registered_model = Model(ws, model_name)
    except:
        logging.info("There is no registered model with specified name available. Please register the specified model first!")

    deployment_config = AciWebservice.deploy_configuration(cpu_cores=2, memory_gb=4, auth_enabled=True)
    service = registered_model.deploy(ws, service_name, [registered_model], inference_config, deployment_config, overwrite=True)
    service.wait_for_deployment(show_output=True)
    logging.info(service.state)
    logging.info("Deployment done!")


def main():
    parser = argparse.ArgumentParser(description="Process input arguments.")
    parser.add_argument("-mn", "--model_name", type=str, help="Name of the model")
    parser.add_argument("-sn", "--service_name", type=str, help="Name of the deployed service")

    args = parser.parse_args()

    model_name = args.model_name
    service_name = args.service_name

    # Directory Setup
    env_dir = "./env"

    # Name of the entry script
    entry_script_name = './entry_script.py'


    # Get the default workspace
    ws, _ = get_current_workspace()
    print(f"Current workspace: {ws}")

    # Deploy the specified model
    deploy_model(ws, model_name, service_name, entry_script_name, env_dir)

if __name__ == '__main__':
    main()

