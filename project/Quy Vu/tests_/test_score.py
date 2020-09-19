import json
import requests
from src.utils import load_metadata


def get_prediction(case):
    metadata = load_metadata()
    uri = metadata['webservice_uri']
    input_json = json.dumps(json.load(open('./tests_/test_data/{}.json'.format(case))))
    response = requests.post(uri, input_json, headers={'Content-Type':'application/json'})
    return json.loads(response.content)


def test_single_case():
    assert get_prediction(case='score_single_case') == [0.5800990636602151]


def test_multiple_case():
    assert get_prediction('score_multiple_case') == [0.5800990636602151, 0.5800990636602151]
