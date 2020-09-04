"""Initialise workspace and environment"""
import json
from azureml.core.environment import CondaDependencies, Environment
from azureml.core import Workspace


def dep_from_pkg_list(pkg_list):
    """Get conda dependencies from list of packages

    Parameters
    ----------
    pkg_list : list
        list of conda packages

    Returns
    -------
    CondaDependencies
        collection of conda dependencies
    """
    return CondaDependencies().create(conda_packages=pkg_list)

# Initialise variables
ENV_NAME = 'heart-failure'
run_pkg = ['pip', 'joblib', 'pandas', 'numpy', 'pyodbc', 'scikit-learn']
test_pkg = run_pkg + ['pytest', 'pylint', 'requests']
workspace = Workspace.from_config()

# Create dependencies for running
run_dependencies = dep_from_pkg_list(run_pkg)
run_dependencies.save('./conda_env.yaml')

# Push to environment
env = Environment.from_conda_specification(ENV_NAME, './conda_env.yaml')
env.python.conda_dependencies=run_dependencies
env.register(workspace=workspace)

# Create dependencies for testing
test_dependencies = dep_from_pkg_list(test_pkg)
test_dependencies.save('./conda_env_test.yaml')

# Initialise metadata
with open('./metadata.json', 'w') as f:
    json.dump({'env_name': ENV_NAME}, f)
