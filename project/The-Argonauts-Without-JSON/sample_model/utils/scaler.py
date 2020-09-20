from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

class Scaler:
    def __init__(self, scaler_type="standard"):
        self.scaler_type = scaler_type
        self.scaler = None
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        if self.scaler is None:
            raise ValueError("Please enter a proper scaler type.")

    def fit(self, arr):
        self.scaler.fit(arr)
        
    def transform(self, arr):
        return self.scaler.transform(arr)

if __name__ == "__main__":
    # Unit test
    arr = np.random.randint(10000, size=(100,10))
    arr.reshape(100,10)
    print(arr)
    scaler = Scaler(scaler_type="minmax")
    scaler.fit(arr)
    arr = scaler.transform(arr)
    print(arr)

        