from azureml.core import Workspace, Datastore
from utils.env_variables import Env


def get_datastore(workspace: Workspace):
    e = Env()
    try:
        if e.ds_datastore_name in workspace.datastores:
            datastore = workspace.datastores[e.ds_datastore_name]
            print(f"Found existing datastore with name: {e.ds_datastore_name}")
        else:
            datastore = Datastore.register_azure_blob_container(
                workspace=workspace,
                datastore_name=e.ds_datastore_name,
                account_name=e.ds_account_name,
                container_name=e.ds_container_name,
                account_key=e.ds_account_key
            )
            datastore.set_as_default()
            print(f"Ragistered datastore with name {e.ds_datastore_name} as default")
        return datastore
    except Exception as ex:
        print(ex)
        print("An error occured trying to attach datastore")
        exit(1)