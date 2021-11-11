from azureml.core import Workspace
from azureml.core.compute import AmlCompute
from azureml.core.compute import ComputeInstance, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from utils.env_variables import Env


def get_compute(workspace: Workspace):
    e = Env()
    try:
        if e.cl_cluster_name in workspace.compute_targets:
            compute_target = workspace.compute_targets[e.cl_cluster_name]
            if compute_target and type(compute_target) is AmlCompute:
                print("Found existing compute target " + e.cl_cluster_name + " so using it.")
        else:
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=e.cl_vm_size,
                vm_priority=e.cl_vm_priority,
                min_nodes=e.cl_min_nodes,
                max_nodes=e.cl_max_nodes,
                idle_seconds_before_scaledown=e.cl_idle_seconds_before_scaledown
            )
            compute_target = ComputeTarget.create(
                workspace, e.cl_cluster_name, compute_config
            )
            compute_target.wait_for_completion(
                show_output=True, min_node_count=None, timeout_in_minutes=10
            )
        return compute_target
    except ComputeTargetException as ex:
        print(ex)
        print("An error occurred trying to provision compute.")
        exit(1)

def get_ml_compute_instance(workspace: Workspace, compute_name):
    try:
        if compute_name in workspace.compute_targets:
            compute_target = workspace.compute_targets[compute_name]
            if compute_target and type(compute_target) is ComputeInstance:
                print("Found existing compute target" + compute_name + " so using it.")
        else: 
            compute_config = ComputeInstance.provisioning_configuration(
                vm_size="STANDARD_D16_V3",
                ssh_public_access=False,
            )
            compute_target = ComputeTarget.create(
                workspace, compute_name, compute_config
            )
            compute_target.wait_for_completion(
                show_output=True, min_node_count=None, timeout_in_minutes=10
            )
            return compute_target
    except ComputeTargetException as ex:
        print(ex)
        print("An error occurred trying to provision compute")
        exit(1)

def get_compute_by_args(workspace: Workspace, cluster_name, vm_size, min_nodes, max_nodes):
    e = Env()
    try:
        if cluster_name in workspace.compute_targets:
            compute_target = workspace.compute_targets[cluster_name]
            if compute_target and type(compute_target) is AmlCompute:
                print("Found existing compute target " + cluster_name + " so using it.")
        else:
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                vm_priority="LowPriority",
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                idle_seconds_before_scaledown=1200
            )
            compute_target = ComputeTarget.create(
                workspace, cluster_name, compute_config
            )
            compute_target.wait_for_completion(
                show_output=True, min_node_count=None, timeout_in_minutes=10
            )
        return compute_target
    except ComputeTargetException as ex:
        print(ex)
        print("An error occurred trying to provision compute.")
        exit(1)