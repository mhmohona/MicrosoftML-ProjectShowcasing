"""
Get data from SQL database, split and export to csv for training and testing
"""
from sklearn.model_selection import train_test_split
from src.utils import DBConnection

# Get data from database
db = DBConnection(source_file='./login_details.json')
all_data = db.query('select * from dbo.patients')

# Split data
LABEL_COL = 'DEATH_EVENT'
features = all_data.drop([LABEL_COL], axis=1)
labels = all_data[LABEL_COL]

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels,
    train_size=0.8, random_state=24, stratify=labels
)

# Export
features_train.to_csv('./data/train/features.csv', index=False)
features_test.to_csv('./data/test/features.csv', index=False)
labels_train.to_csv('./data/train/labels.csv', index=False)
labels_test.to_csv('./data/test/labels.csv', index=False)
