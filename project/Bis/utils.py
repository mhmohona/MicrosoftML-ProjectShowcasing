import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from IPython.display import display
from sklearn.metrics import classification_report, confusion_matrix
from BaselineRemoval import BaselineRemoval
from scipy.signal import welch, find_peaks, savgol_filter

sns.set_style("whitegrid")
plt.style.use('ggplot')
sns.set(font_scale=1.5)
sns.set_palette("bright")


def load_data(train_folder, test_folder, train_labels, test_labels):
    """
    loads train and test signals and labels
    input: train and test signal and label folder in .txt form
    returns numpy ndarray of train, test signals and labels
    """
    train_pathlist = sorted(Path(train_folder).rglob('*.txt'))
    test_pathlist = sorted(Path(test_folder).rglob('*.txt'))

    train_signals = [np.loadtxt(path) for path in train_pathlist]
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))

    test_signals = [np.loadtxt(path) for path in test_pathlist]
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))

    train_labels = np.loadtxt(Path(train_labels))
    test_labels = np.loadtxt(Path(test_labels))

    print("number of train signals, length of each signal, number of components:", train_signals.shape)
    print("number of test signals, length of each signal, number of components:", test_signals.shape)
    return train_signals, test_signals, train_labels, test_labels


def get_classification_results(models, X_train, X_test, y_train, y_test):
    
    """
    input: Dictionary of ML models with Gridsearch parameters.
    Prints classification reports and confusion matrices.
   """
    
    for i in models:
        models[i].fit(X_train, y_train)
        print(i, "Report:")
        print("############################################")
        print("Train Accuracy: {}".format(models[i].score(X_train, y_train)))
        print("Test Accuracy : {}".format(models[i].score(X_test, y_test)))
        y_pred = models[i].predict(X_test)
        best_params = models[i].best_params_
        best_params_df = pd.DataFrame(best_params, index=[0])
        print (" ")
        print("Best parameters:")
        display(best_params_df)
        report = classification_report(y_test, y_pred, output_dict=True)
        df = pd.DataFrame(report).transpose()
        print (" ")
        print("Classification Report:")
        display(df)
        cf_matrix = confusion_matrix(y_test, y_pred)
        print (" ")
        print("Confusion Matrix:")
        df1 = pd.DataFrame(cf_matrix)
        display(df1)
        cf_matrix_normalized_p = cf_matrix / cf_matrix.astype(np.float).sum(axis=0) 
        cf_matrix_normalized_r = cf_matrix / cf_matrix.astype(np.float).sum(axis=1)
        plt.figure(figsize=[8, 6])
        print("Normalized precision cf")
        sns.heatmap(cf_matrix_normalized_p, annot=True, cmap='Blues')
        plt.show()
        plt.figure(figsize=[8, 6])
        print("Normalized recall cf")
        sns.heatmap(cf_matrix_normalized_r, annot=True, cmap='Blues')
        plt.show()
        

def train_class(model, num_epochs, bs, train_loader, test_loader, optimizer, criterion):
    
   """
   Trains a CNN classifier and prints train and test results.
   input: CNN model, number of epochs, batch size, PyTorch train loader and test loaders, optimizer and loss function
   returns a Pandas dataframe with loss values, plots training and test losses
   """

    loss_values = {
        'train': [],
        'test': []
    }

    accuracy_values = {
        'train': [],
        'test': []
    }

    final_loss_df = pd.DataFrame()
    final_acc_df = pd.DataFrame()

    for epoch in range(num_epochs):
        start = timer()
        model.train()
        avg_training_loss = 0.
        correct = 0.

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = Variable(inputs), Variable(labels)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_training_loss += loss.item() / len(train_loader)
#             correct += (outputs == labels).float().sum()
        loss_values['train'].append(avg_training_loss)
#         accuracy_values['train'].append((100 * correct // len(X_train_tensor)))

        model.eval()
        avg_test_loss = 0.
        correct = 0.

        for i, (inputs, labels) in enumerate(test_loader):

            inputs, labels = Variable(inputs), Variable(labels)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, labels) # Calculate the loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            avg_test_loss += loss.item() / len(train_loader)
#             correct += (outputs == labels).float().sum()
        loss_values['test'].append(avg_test_loss)
#         accuracy_values['train'].append((100 * correct // len(X_train_tensor)))

        end = timer()
        elapsed_time = end - start

        if epoch % 10 == 1:
            print('Epoch {}/{} \t loss={:.4f} \t test_loss={:.4f} \t time={:.2f}s'.format(
                epoch + 1, num_epochs, avg_training_loss, avg_test_loss, elapsed_time))
#     print(loss_values)
    loss_df = pd.DataFrame.from_dict(loss_values)
#     acc_df = pd.DataFrame.from_dict(accuracy_values)
    final_loss_df = final_loss_df.append(loss_df)
#     final_acc_df = final_acc_df.append(acc_df)
    final_loss_df['epoch'] = final_loss_df.index
    final_loss_df = final_loss_df.groupby('epoch').mean().reset_index()

    print('average fold losses: \n', final_loss_df)
#     print('average fold losses: \n', final_acc_df)
    plt.plot(final_loss_df.epoch, final_loss_df.train, label='train')
    plt.plot(final_loss_df.epoch, final_loss_df.test, label='test')
    plt.legend()
    plt.show()
    plt.savefig('train_test_plot.jpg')
            
        
def get_cnn_cf_matrix(model, X_test_tensor, test_labels):
    
    """
    input: CNN model, test data in tensor form, test labels.
    Prints classification reports and confusion matrices.
    """
    
    with torch.no_grad():
        output = model(X_test_tensor)
        pred = np.argmax(output, axis=1) + 1
        report = classification_report(pred, test_labels, output_dict=True)
        df = pd.DataFrame(report).transpose()
        display(df)
        cf_matrix = confusion_matrix(pred, test_labels)
        df1 = pd.DataFrame(cf_matrix)
        print("Number of samples in each class in test set:", np.unique(test_labels, return_counts=True))
        display(df1)
        plt.figure(figsize=[8, 6])
        cf_matrix_normalized_p = cf_matrix / cf_matrix.astype(np.float).sum(axis=0) 
        cf_matrix_normalized_r = cf_matrix / cf_matrix.astype(np.float).sum(axis=1)
        print("Normalized precision cf:")
        sns.heatmap(cf_matrix_normalized_p, annot=True, cmap='Blues')
        plt.show()
        plt.figure(figsize=[8, 6])
        print("Normalized recall cf:")
        sns.heatmap(cf_matrix_normalized_r, annot=True, cmap='Blues')
        plt.show()


def plot_stem (x, y, title):
    """
    input: frequencies and intensities as lists (x, y). title is user defined title
    returns stem plot
    """
    plt.figure(figsize=[8, 6])
    plt.stem(x, y, use_line_collection=True)
    plt.xlabel('time/Sec')
    plt.ylabel('amplitude')
    plt.title(title)
    plt.plot(x, y)
    plt.show()


def plot_feature(x, y, feature):
    """
    input: frequencies and intensities as lists (x, y). title is user defined title for features, FFT, PSD, etc.
    returns baseline smoothed spectrum with peak detection
    """
    baseObj=BaselineRemoval(y)
    y=baseObj.ModPoly(2)
    y = savgol_filter(y, 5, 2)
    peaks, _ = find_peaks(y)
    first_n_freq = x[peaks][np.argsort(-y[peaks])][0:5]
    first_n_int =  y[peaks][np.argsort(-y[peaks])][0:5]
    plt.plot(x, y)
    plt.plot(x[peaks], y[peaks], "x")
    plt.xlabel('Freq/Hz')
    plt.ylabel('amplitude')
    plt.title(feature)
    plt.show()    
