from sklearn.model_selection import train_test_split
import numpy as np

def split_data(X, y, test_size=0.1, random_state=42, shuffle=True):
   return train_test_split(X,
                           y, 
                           test_size=test_size, 
                           random_state=random_state,
                           shuffle=shuffle)

if __name__ == "__main__":
    arr = np.random.randn(100,10)
    X_train, X_test, y_train, y_test = split_data(arr[:,:-1], arr[:,-1])
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)