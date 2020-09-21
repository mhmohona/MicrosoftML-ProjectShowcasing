import re
import json
import requests

import numpy as np

URL = 'http://localhost:5000/getkeypts'


def get_keypts(image_url: str, silent: bool, method: str='docker') -> list:
    """
    Calls the Facial Landmark API to calculate the points.

    Args:
        image_url (str): the path of the image
        silent (bool): if turned on then error messages are not printed
        method (str): the method to use to calculate the facial points
    
    Returns:
        keypts (list): the list containing all the keypoints

    Raises:
        ValueError: if the method selected is not available

    """
    if method == 'docker':
        f = {'file': open(image_url, 'rb').read()}
        result = requests.post(URL, files=f)
        if result.status_code == 200:
            try:
                keypts = json.loads(result.text)['points']
                keypts = np.array(keypts).reshape((1,-1))[0]
                return keypts
            except:
                return None
        else:
            if not silent: 
                print(result.text)
            return None
    else:
        raise ValueError("method not valid")