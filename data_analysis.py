import tensorflow as tf
from tensorflow import keras
import librosa
import os
from glob import glob
import numpy as np
import pandas as pd
import random
import pickle
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, LSTM, Dense

import warnings
warnings.filterwarnings("ignore")

def explore():
    df = pd.read_csv("data/train_curated.csv")
    df1 = pd.read_csv("./data/sample_submission.csv")
    labels = list(df1.columns)[1:]
    for l in labels:
        df[l] = 0
    
    audios = glob(os.path.join("data","train_curated/*.wav"))
    # audios[:10]
    fnames = [Path(f).name for f in audios]
    def set_label(x):
        labels = x.strip("\"").split(",")
        return labels

    df["labels"] = df["labels"].apply(lambda x: set_label(x))

    for index, row in df.iterrows():
        lbls = row["labels"]
        for l in lbls:
            df.loc[index,l] = 1
    features = []
    fnames = list(df["fname"])
    for fname in tqdm(fnames):
        features.append(extract_features(fname))
    # features = np.array(features)
    np.save('freeaudio_features_trimmed.npy',features)

def add_padding(features, mfcc_max_padding=1000):
    padded = []

    # Add padding
    for i in range(len(features)):
        px = features[i]
        size = len(px)
        # Add padding if required
        if (size < mfcc_max_padding):
            xDiff = mfcc_max_padding - size
            xLeft = xDiff//2
            xRight = xDiff-xLeft
            px = np.pad(px, pad_width=((xLeft, xRight)), mode='constant')
        elif (size > mfcc_max_padding):
            px = px[:mfcc_max_padding]
        padded.append(px)

    return np.array(padded)

def cut_audio(x,sr):
    duration = 10
    input_length = sr*duration
    if len(x) > input_length:
        max_offset = len(x) - input_length
        offset = np.random.randint(max_offset)
        x = x[offset:(input_length+offset)]
    else:
        if input_length > len(x):
            max_offset = input_length - len(x)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        x = np.pad(x, (offset, input_length - len(x) - offset), "constant")
    return x
    
def extract_features(fname):
    audio_path = f"data/train_curated/{fname}"
    x,sr = librosa.load(audio_path,sr=None)
    x,_ = librosa.effects.trim(x,top_db=60)
    x = cut_audio(x,sr)
    mfcc = librosa.feature.mfcc(y=x,sr=sr,n_mfcc=20)
    mel = librosa.feature.melspectrogram(y=x,sr=sr,n_mels=30,fmin=20,n_fft=512)
    chroma_stft = librosa.feature.chroma_stft(y=x,sr=sr,n_chroma=12)
    tonnetz = librosa.feature.tonnetz(y=x,sr=sr,chroma=chroma_stft)
    features = (mfcc,chroma_stft,mel,tonnetz)
    features = np.concatenate(features,axis=0)
    # features = add_padding(features)
    return features

if __name__ == "__main__":
    explore()

