"""
Created in November 2021

Python code to construct the pipeline for model registration

@author: Martin Danner
@company: scieneers GmbH
@mail: martin.danner@scieneers.de
"""


from utils.attach_compute import get_compute_by_args
from utils.attach_datastore import get_datastore
from utils.env_variables import Env
from azureml.core import Environment, Experiment, Workspace
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import Pipeline, PipelineEndpoint, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.authentication import InteractiveLoginAuthentication
import os


# Read the .env variables
e = Env()

# Directory Setup
base_dir = os.path.dirname(__file__)
script_dir = os.path.join(base_dir, "scripts")
env_dir = os.path.join(script_dir, "env")

# Get the workspace
try:
    interactive_auth  = InteractiveLoginAuthentication(tenant_id=e.tenant_id)
    ws = Workspace.get(
        name=e.ws_name,
        subscription_id=e.ws_subscription_id,
        resource_group=e.ws_resource_group,
        auth=interactive_auth
    )
    print("get_workspace:")
    print(ws)

except:
    print("Workspace not found!")

cluster_name = "cl-rundstedt-ml"
vm_size = "STANDARD_NC6S_V3"

# Get Azure machine learning cluster
aml_compute = get_compute_by_args(ws, cluster_name, vm_size, 0, 2)
if aml_compute is not None:
    print("aml_compute:")
    print(aml_compute)

# Set environment
env = Environment.get(workspace=ws, name="AzureML-Minimal")

print(os.getcwd())
print(__file__)

# Create Environment from yaml file
env_filepath = os.path.join(env_dir, "cl_env.yml")
env = Environment.from_conda_specification(
    name="Cluster_Env", file_path=env_filepath)

# create a new runconfig object
run_config = RunConfiguration()
run_config.target = aml_compute
run_config.environment = env

# Setup pipeline parameters
model_name = PipelineParameter(
    name="model_name", default_value="Test_Bert_Model_IMBD")
model_file_path_on_blob = PipelineParameter(
    name="model_file_path_on_blob", default_value="models/Test_Bert_Model_IMBD")
tag = PipelineParameter(
    name="tag", default_value="MyCustomTag")
datastore_name = PipelineParameter(
    name="datastore_name", default_value="datalake_rundstedt")

# Setup python script task
source_directory = os.path.join(base_dir)
step1 = PythonScriptStep(
    runconfig=run_config,
    allow_reuse=False,
    name="Registration_Step",
    script_name="model_registration_script.py",
    source_directory=script_dir,
    arguments=["--model_name", model_name, "--model_file_path_on_blob", model_file_path_on_blob, "--tag", tag,  "--datastore_name", datastore_name]
    )

# Set the pipeline name
pipeline_name = "Register_Model"

# Set the Experiment name
experiment_name = "Pipeline_Deployment"

# Build the pipeline
pipeline = Pipeline(workspace=ws, steps=[step1])

# Execute the pipeline
pipeline_run = Experiment(ws, experiment_name).submit(pipeline)

pipeline_run.wait_for_completion()

#Publish the pipeline
published_pipeline = pipeline_run.publish_pipeline(
    name=pipeline_name, continue_on_step_failure=True, description=pipeline_name, version="1.0")

# Create Endpoint if it doesnt exist, otherwise add published pipeline as new default
try: 
    pipeline_endpoint = PipelineEndpoint.get(ws, name=pipeline_name)
    pipeline_endpoint.add_default(published_pipeline)

except:
    pipeline_endpoint = PipelineEndpoint.publish(
        ws, name=pipeline_name, pipeline=published_pipeline, description=pipeline_name)