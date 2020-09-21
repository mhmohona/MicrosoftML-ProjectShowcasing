import json
import requests


class HeartFailurePredictor:
    def __init__(self, expected_len):
        self.expected_len = expected_len

    def check_user_input_len(self, user_input):
        assert len(user_input) == self.expected_len,\
            'Expect input for {} patient(s). Please re-enter'.format(self.expected_len)

    def request_user_input(self, msg):
        user_input = [float(item) for item in input(msg).split(',')]
        try:
            self.check_user_input_len(user_input)
        except AssertionError as e:
            print(e)
            user_input = self.request_user_input(msg)

        return user_input


def get_prediction(input_json):
    metadata = json.load(open('./metadata.json'))
    uri = metadata['webservice_uri']
    response = requests.post(uri, input_json, headers={'Content-Type':'application/json'})
    return json.loads(response.content)


if __name__ == '__main__':
    print('Welcome to the Heart Failure Predictor')
    print('It uses machine learning to determine the probability of a patient suffering from heart failure.')

    # Requesting inputs
    predictor = HeartFailurePredictor(int(input('Number of patient(s): ')))
    if predictor.expected_len > 1:
        print('Please enter patients\'s data separated by comma')
    age = predictor.request_user_input('age (Years): ')
    anaemia = predictor.request_user_input('anaemia (0 for no, 1 for yes): ')
    high_blood_pressure = predictor.request_user_input('high_blood_pressure (0 for no, 1 for yes): ')
    diabetes = predictor.request_user_input('diabetes (0 for no, 1 for yes): ')
    smoking = predictor.request_user_input('smoking (0 for no, 1 for yes): ')
    sex = predictor.request_user_input('sex (0 for woman, 1 for man): ')
    creatinine_phosphokinase = predictor.request_user_input('creatinine_phosphokinase (mcg/L): ')
    ejection_fraction = predictor.request_user_input('ejection_fraction at each contraction (percentage): ')
    platelets = predictor.request_user_input('platelets (kiloplatelets/mL): ')
    serum_creatinine = predictor.request_user_input('serum_creatinine (mg/dL): ')
    serum_sodium = predictor.request_user_input('serum_sodium (mEq/L): ')

    # Build json payload
    input_json = json.dumps({
        'age': age,
        'anaemia': anaemia,
        'creatinine_phosphokinase': creatinine_phosphokinase,
        'diabetes': diabetes,
        'ejection_fraction': ejection_fraction,
        'high_blood_pressure': high_blood_pressure,
        'platelets': platelets,
        'serum_creatinine': serum_creatinine,
        'serum_sodium': serum_sodium,
        'sex': sex,
        'smoking': smoking
    })

    # Get prediction
    prediction = [round(pred, 2) for pred in get_prediction(input_json)]

    if predictor.expected_len > 1:
        print('The patients\'s probabilities of heart failure respectively are: {}'.format(str(prediction)[1:-1]))
    else:
        print('The patient\'s probability of heart failure is: {}'.format(str(prediction)[1:-1]))
    input('Press Enter to exit')
