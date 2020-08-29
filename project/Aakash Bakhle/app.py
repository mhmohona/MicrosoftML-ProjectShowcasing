import os
import torch

from fastai.vision.all import *
import pretrainedmodels

import numpy as np
import pandas as pd

import streamlit as st
from pathlib import Path

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

path = Path(__file__).parent

st.set_option('deprecation.showfileUploaderEncoding', False)

def se_resnext50_32x4d(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))

def predict(image, model):
    test_image = Image.open(image).convert("RGB").resize((224, 224)) #resize the input image
    test_image = np.array(test_image)
    model = MODEL
    pred, pred_idx, probabs = model.predict(test_image)

    return pred, pred_idx, probabs

def get_k_probs(probabs, k):
    topk_labels = MODEL.dls.vocab[probabs.topk(k)[1]]
    topk_probs = probabs.topk(k)[0]

    topk_probs = topk_probs.numpy()*100
    topk_labels = np.array(topk_labels)

    plot_fn(topk_probs, topk_labels)

def stream_lit():
    st.title("Car Model and Make Detector")

    label_df = pd.read_csv(path/'labels.csv')
    label_df.drop('Unnamed: 0', axis = 1, inplace = True)
    label_df.index = range(1,197)
    st.sidebar.table(label_df)
    file_image = st.file_uploader("Upload a car photo", type = ['jpeg', 'jpg', 'png'])

    if file_image is None:
        pass
        # st.write("upload image")

    else :
        # input_img = open_image(file_image)
        pred, pred_idx, probabs = predict(file_image, MODEL)
        st.write("Input photo")
        st.image(file_image, use_column_width = True)

        st.write("Prediction :")
        st.write(pred)

        st.write('Probability:')
        st.write(f'{probabs[pred_idx]:.04f}')

        k = st.slider('top k probabilities',1,5, value = 2)
        get_k_probs(probabs, k)

def plot_fn(topk_p, topk_l):
    fig, ax = plt.subplots()
    ax.barh(topk_l, topk_p)
    ax.set_xticks(range(0,110,10))
    ax.invert_yaxis()
    plt.xlabel('% probablity')
    plt.ylabel('Car name')

    st.pyplot()

    create_table(topk_p, topk_l)

def create_table(topk_p, topk_l):
    df = pd.DataFrame()
    df['car names'] = topk_l
    df['probability'] = topk_p

    st.dataframe(df)

if __name__ == '__main__':
    MODEL = load_learner(path/'mymodel.pkl','cpu')
    stream_lit()