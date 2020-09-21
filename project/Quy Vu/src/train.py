"""Model training script"""
import json
import joblib
import numpy as np
from azureml.core import Workspace, Experiment
from azureml.core.model import Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, recall_score
from sklearn.model_selection import train_test_split
from src.utils import DBConnection, update_metadata

# Get data from database
db = DBConnection(source_file='./login_details.json')
all_data = db.query('select * from dbo.patients')

# Split data
LABEL_COL = 'DEATH_EVENT'
features = all_data.drop([LABEL_COL, 'time'], axis=1)
labels = all_data[LABEL_COL]

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels,
    train_size=0.8, random_state=24, stratify=labels
)

# Initialise variables for model training
max_depths = np.arange(2, 8)
min_samples_splits = [2, 4, 8, 16, 32]
BEST_RECALL = 0
MODEL_FILE = 'best_model.pkl'
MODEL_PATH = './models/' + MODEL_FILE
MODEL_NAME = EXPERIMENT_NAME = 'heartfailure_randomforest_greedy'
MODEL_DESCRIPTION = 'Heart failure predictor with RF'
workspace = Workspace.from_config()

# Create new experiment
experiment = Experiment(workspace=workspace, name=EXPERIMENT_NAME)

# Grid search
for max_depth in max_depths:
    for min_samples_split in min_samples_splits:
        # Log
        run = experiment.start_logging()
        run.log('max_depth', max_depth)
        run.log('min_samples_split', min_samples_split)

        # Train
        model = RandomForestClassifier(
            min_samples_split=min_samples_split,
            max_depth=max_depth,
            n_estimators=100
        )

        model.fit(features_train, labels_train)
        labels_pred_test = model.predict(features_test)
        labels_pred_test_prob = model.predict_proba(features_test)[:, 1]

        # Log
        run.log('accuracy', accuracy_score(labels_test, labels_pred_test))
        run.log('auc', roc_auc_score(labels_test, labels_pred_test_prob))
        run.log('log_loss', log_loss(labels_test, labels_pred_test_prob))
        current_recall = recall_score(labels_test, labels_pred_test)
        run.log('recall', current_recall)

        # Save best
        if current_recall > BEST_RECALL:
            BEST_RECALL = current_recall
            run_metrics = run.get_metrics()

            joblib.dump(value=model, filename=MODEL_PATH)
            run.upload_file(name=MODEL_FILE, path_or_stream=MODEL_PATH)

        run.complete()

with open('./outputs/best_run_metrics.json', 'w') as f:
    json.dump(run_metrics, f, indent=4)

# Register model
model = Model.register(
    model_path=MODEL_PATH,
    model_name=MODEL_NAME,
    description=MODEL_DESCRIPTION,
    workspace=workspace
)

# Update metadata
update_metadata({
    "model_file": MODEL_FILE,
    "model_path": MODEL_PATH,
    "experiment_name": EXPERIMENT_NAME,
    "model_name": MODEL_NAME,
    "model_description": MODEL_DESCRIPTION
})
