"""Env dataclass to load and hold all environment variables
"""
from dataclasses import dataclass
import os
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Env:
    """Loads all environment variables into a predefined set of properties
    """

    # to load .env file into environment variables for local execution
    load_dotenv()

    # Azure ML Tenant ID
    tenant_id: Optional[str] = os.environ.get("TENANT_ID")

    # Azure ML Workspace Variables
    ws_name: Optional[str] = os.environ.get("WS_NAME")
    ws_subscription_id: Optional[str] = os.environ.get("WS_SUBSCRIPTION_ID")
    ws_resource_group: Optional[str] = os.environ.get("WS_RESOURCE_GROUP")
    ws_location: Optional[str] = os.environ.get("WS_LOCATION")

    # Azure ML Cluster Variables
    cl_cluster_name: Optional[str] = os.environ.get("CL_CLUSTER_NAME")
    cl_vm_size: Optional[str] = os.environ.get("CL_VM_SIZE")
    cl_vm_priority: Optional[str] = os.environ.get("CL_VM_PRIORITY")
    cl_min_nodes: int = int(os.environ.get("CL_MIN_NODES", 0))
    cl_max_nodes: int = int(os.environ.get("CL_MAX_NODES", 20))
    cl_idle_seconds_before_scaledown: int = int(os.environ.get("CL_IDLE_SECONDS_BEFORE_SCALEDOWN", 1200))

    # Azure ML Datastore Variables
    ds_account_name: Optional[str] = os.environ.get("DS_ACCOUNT_NAME")
    ds_container_name: Optional[str] = os.environ.get("DS_CONTAINER_NAME")
    ds_datastore_name: Optional[str] = os.environ.get("DS_DATASTORE_NAME")
    ds_account_key: Optional[str] = os.environ.get("DS_ACCOUNT_KEY")

    # Azure ML Service Connections
    sc_azureml: Optional[str] = os.environ.get("SC_AZUREML")
