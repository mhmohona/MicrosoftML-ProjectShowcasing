# pylint: disable=abstract-class-instantiated
# Used to silence the pylint error message for deleting a webservice
"""Deploy model to azure"""
from azureml.core import Workspace, Environment, Webservice
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from azureml.exceptions import WebserviceException
from src.utils import update_metadata, load_metadata

# Initialise
METADATA = load_metadata()
SERVICE_DESCRIPTION = 'Heart failure predictor web service'
SERVICE_NAME = 'heartfailure-prediction'
CPU_CORES = 1
MEMORY_GB = 1

# Get environment
workspace = Workspace.from_config()
environment = Environment.get(workspace=workspace, name=METADATA['env_name'])

# Deploy container
inference_config = InferenceConfig(
    entry_script='./src/score.py', environment=environment,
)
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=CPU_CORES, memory_gb=MEMORY_GB,
    description=SERVICE_DESCRIPTION
)

# Deploy as web service
try:
    # Remove any existing service under the same name.
    Webservice(workspace, SERVICE_NAME).delete()
except WebserviceException:
    pass

model = Model(workspace, METADATA['model_name'])

webservice = Model.deploy(
    workspace=workspace,
    name=SERVICE_NAME,
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config,
    overwrite=True
)

webservice.wait_for_deployment(show_output=True)

webservice.get_logs()

# Update metadata
update_metadata({
    "service_description": SERVICE_DESCRIPTION,
    "service_name": SERVICE_NAME,
    "cpu_cores": CPU_CORES,
    "memory_gb": MEMORY_GB,
    'webservice_uri': webservice.scoring_uri
})
