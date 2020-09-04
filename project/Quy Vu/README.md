# Initialise environment

```bash
python -m venv .venv
pip install -r requirements.txt
```

Create dev environment

```bash
conda env create -f conda_env_test.yaml --prefix ./envs_dev
```

Update dev environment: Run the initialise script first then the following bash command

```bash
conda env update --prefix ./envs_dev --file conda_env_test.yaml
```
