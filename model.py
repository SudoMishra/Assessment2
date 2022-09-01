from pickletools import optimize
from re import X
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
import sklearn

import matplotlib.pyplot as plt
import seaborn as sns
from keras.layers import Input, LSTMCell, Dense, Conv2D, MaxPool2D, Dropout, Flatten, GlobalMaxPool2D, BatchNormalization, RNN
from skmultilearn.model_selection import IterativeStratification
from sklearn.neural_network import MLPClassifier
from keras.applications.vgg19 import VGG19
from sklearn.metrics import multilabel_confusion_matrix, ConfusionMatrixDisplay, confusion_matrix

# k_fold = IterativeStratification(n_splits=2, order=1):
# for train, test in k_fold.split(X, y):
#     classifier.fit(X[train], y[train])
#     result = classifier.predict(X[test])

MAX_SEQ_LENGTH = 862
embed_dim = 68
num_class = 80
n_epochs = 15
batch_size = 64
embed_start = 0
embed_end = embed_start+embed_dim
def _one_sample_positive_class_precisions(scores, truth):
  """Calculate precisions for each true class for a single sample.
  
  Args:
    scores: np.array of (num_classes,) giving the individual classifier scores.
    truth: np.array of (num_classes,) bools indicating which classes are true.

  Returns:
    pos_class_indices: np.array of indices of the true classes for this sample.
    pos_class_precisions: np.array of precisions corresponding to each of those
      classes.
  """
  num_classes = scores.shape[0]
  pos_class_indices = np.flatnonzero(truth > 0)
  # Only calculate precisions if there are some true classes.
  if not len(pos_class_indices):
    return pos_class_indices, np.zeros(0)
  # Retrieval list of classes for this sample. 
  retrieved_classes = np.argsort(scores)[::-1]
  # class_rankings[top_scoring_class_index] == 0 etc.
  class_rankings = np.zeros(num_classes, dtype=np.int)
  class_rankings[retrieved_classes] = range(num_classes)
  # Which of these is a true label?
  retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
  retrieved_class_true[class_rankings[pos_class_indices]] = True
  # Num hits for every truncated retrieval list.
  retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
  # Precision of retrieval list truncated at each hit, in order of pos_labels.
  precision_at_hits = (
      retrieved_cumulative_hits[class_rankings[pos_class_indices]] / 
      (1 + class_rankings[pos_class_indices].astype(np.float)))
  return pos_class_indices, precision_at_hits

# Calculate the overall lwlrap using sklearn.metrics function.

def calculate_overall_lwlrap_sklearn(truth, scores):
  """Calculate the overall lwlrap using sklearn.metrics.lrap."""
  # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
  sample_weight = np.sum(truth > 0, axis=1)
#   nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)

  overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
      truth, 
      scores, 
      sample_weight=sample_weight)
  return overall_lwlrap

def lwlwrap_tf(truth,scores):
    return tf.py_function(calculate_overall_lwlrap_sklearn,(truth,scores),tf.double)  

def get_data():
    features = np.load('freeaudio_features_trimmed.npy').reshape(-1,862,68)
    # print(features.shape)
    # exit()
    df = pd.read_csv("data/train_curated.csv")
    df1 = pd.read_csv("./data/sample_submission.csv")
    labels = list(df1.columns)[1:]
    for l in labels:
        df[l] = 0
    
    # audios = glob(os.path.join("data","train_curated/*.wav"))
    # audios[:10]

    # fnames = [Path(f).name for f in audios]
    fnames = list(df["fname"])
    def set_label(x):
        labels = x.strip("\"").split(",")
        return labels

    df["labels"] = df["labels"].apply(lambda x: set_label(x))

    for index, row in df.iterrows():
        lbls = row["labels"]
        for l in lbls:
            df.loc[index,l] = 1
    
    y_train = df.iloc[:,2:].to_numpy()
    x_train = features[:,0:MAX_SEQ_LENGTH,embed_start:embed_end]

    return x_train,y_train

def get_model():
    inp_seq = Input(shape=(MAX_SEQ_LENGTH,embed_dim))
    x = RNN(LSTMCell(256,dropout=0.2),)(inp_seq)
    # x = RNN(LSTMCell(128,dropout=0.2))(x)
    # x = Dense(128,activation='relu')(x)
    # x = Dropout(0.2)(x)
    # x = Dense(64,activation="relu")(x)
    
    y = Dense(80,activation='sigmoid')(x)
    model = keras.Model(inputs=inp_seq,outputs=y,name="RNN_Model")
    return model

def fusion_model():
    inp_seq_1 = Input(shape = (MAX_SEQ_LENGTH,20))
    inp_seq_2 = Input(shape = (MAX_SEQ_LENGTH,30))
    x1 = RNN(LSTMCell(100,dropout=0.2))(inp_seq_1)
    x2 = RNN(LSTMCell(100,dropout=0.2))(inp_seq_2)
    d1 = Dense(64,activation='relu')(x1)
    d2 = Dense(64,activation='relu')(x2)
    x = keras.layers.Concatenate(axis=1)([d1,d2])
    x = Dense(32,activation='relu')(x)
    y = Dense(80,activation="sigmoid")(x)
    model = keras.Model(inputs=[inp_seq_1,inp_seq_2],outputs=y,name="FusionModel")
    return model

def vgg_fine_tune():
    inp_seq = Input(shape=(MAX_SEQ_LENGTH,embed_dim,3))
    base_model = VGG19(weights="imagenet",include_top=False,input_shape=(MAX_SEQ_LENGTH,embed_dim,3))
    for layer in base_model.layers:
        layer.trainable = False
    # base_model.trainable=False
    x = base_model(inp_seq)
    x = keras.layers.Flatten(name='flatten')(x)
    x = keras.layers.Dense(256, activation='relu', name='fc1')(x)
    # x = keras.layers.Dense(256, activation='relu', name='fc2')(x)
    x = keras.layers.Dense(80, activation='sigmoid', name='predictions')(x)
    model = keras.models.Model(inputs=inp_seq, outputs=x)
    return model

def cnn_model():
    inp_seq = Input(shape=(MAX_SEQ_LENGTH,embed_dim,1))
    x = Conv2D(32,(5,5),strides=1)(inp_seq)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32,(5,5),strides=1,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    # x = GlobalMaxPool2D(keepdims=True)(x)
    x = Conv2D(32,(5,5),strides=1,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    # x = GlobalMaxPool2D(keepdims=True)(x)
    x = Conv2D(32,(5,5),strides=1,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    # x = GlobalMaxPool2D(keepdims=True)(x)
    x = Conv2D(32,(5,5),strides=1,activation='relu')(x)
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    # x = GlobalMaxPool2D(keepdims=True)(x)
    # x = Conv2D(16,(10,2),strides=1,activation='relu')(x)
    # # x = GlobalMaxPool2D(keepdims=True)(x)
    # x = Conv2D(16,(10,2),strides=1,activation='relu')(x)
    # # x = GlobalMaxPool2D(keepdims=True)(x)
    # x = Conv2D(16,(10,2),strides=1,activation='relu')(x)
    # # x = GlobalMaxPool2D(keepdims=True)(x)
    # x = Conv2D(16,(10,2),strides=1,activation='relu')(x)
    x = MaxPool2D(pool_size=(5,5))(x)
    # x = Conv2D(16,(3,3),strides=1,activation='relu')(x)
    # x = MaxPool2D(pool_size=(3,3),strides=(1,1))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    # x = Dense(64,activation='relu')(x)
    # x = Dense(10,activation='softmax')(x)
    # # x = keras.layers.RNN(keras.layers.LSTMCell(50,dropout=0.2))(inp_seq)
    x = Dense(32,activation='relu')(x)
    y = Dense(80,activation='sigmoid')(x)
    model = keras.Model(inputs=inp_seq,outputs=y,name="Model")
    return model


def scheduler(epoch,lr):
    if epoch< int(0.2*n_epochs):
        return lr
    else:
        return 0.99*lr


def train_model(cnn=False):
    x,y = get_data()
    k_fold = IterativeStratification(n_splits=2, order=1)
    models = dict()
    datasets = dict()
    if cnn:
        x = x[..., tf.newaxis].astype("float32")
        x = np.repeat(x,[3],axis=-1)
        print(x.shape)
    feature_dims = [20,30,12,6]
    last = 0
    # x,y = x[:10], y[:10]
    # callbacks = [keras.callbacks.LearningRateScheduler(scheduler)]
    callbacks = [keras.callbacks.ReduceLROnPlateau(
                    monitor="lwlwrap_tf",
                    factor=0.9,
                    patience=4,
                    verbose=0,
                    mode="auto",
                    min_delta=0.0001
                )]
    for idx, (train, test) in enumerate(k_fold.split(x, y)):
        # print(train.shape,test.shape)
        # classifier.fit(X[train], y[train])
        # result = classifier.predict(X[test])
        x_train,y_train,x_val,y_val = x[train],y[train],x[test],y[test]
        print(x_train.shape)
        # x_train,y_train = x_train[:10],y_train[:10]
        model = vgg_fine_tune() if cnn else get_model()
        print(model.summary())
        # for idx, model in enumerate(mods[0:1]):
        model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.BinaryCrossentropy(),metrics=[lwlwrap_tf,"accuracy"])
        # model.evaluate(x_train[:10],y_train[:10])
        # fd = feature_dims[idx]
        # xt,yt,xv,yv = x_train[:,:,last:fd,:],y_train[:,:,last:fd,:],x_val[:,:,last:fd,:],y_val[:,:,last:fd,:]
        # last=fd
        history = model.fit(x_train,y_train,epochs=n_epochs,batch_size=64,validation_data=(x_val,y_val),callbacks=callbacks)
        model.save(f'saved_models/model_dim_{20}')
        # print(history.history["loss"])#,history.history["metrics"])
        # clf = MLPClassifier(random_state=1, max_iter=1000).fit(x_train,y_train)
        # preds = clf.predict(x_val)
        # print(f"lwlwrap: {calculate_overall_lwlrap_sklearn(y_val,preds)}")

        models[idx] = model
        datasets[idx] = (train,test)
        model.save(f".saved_models/model_{idx}.hdf5")
        break
    with open('saved_datasets.pkl', 'wb') as f:
        pickle.dump(datasets, f)
    
def fusion_train():
    x,y = get_data()
    x,y = x[:10], y[:10]
    k_fold = IterativeStratification(n_splits=2, order=1)
    x_1,x_2 = x[:,:,:20], x[:,:,20:]
    datasets = dict()
    for idx, (train,test) in enumerate(k_fold.split(x,y)):
        x_t1,x_t2,y_t,x_v1,x_v2,y_v = x_1[train], x_2[train], y[train], x_1[test], x_2[test], y[test]
        model = fusion_model()
        print(model.summary())
        model.compile(optimizer=keras.optimizers.Adam(),loss=keras.losses.BinaryCrossentropy(),metrics=[lwlwrap_tf])
        history = model.fit(x=[x_t1,x_t2],y=y_t,validation_data=([x_v1,x_v2],y_v),epochs=n_epochs,batch_size=batch_size)
        model.save(f'saved_models/fusion_model')
        datasets[idx] = (train,test)
    with open('saved_datasets.pkl', 'wb') as f:
        pickle.dump(datasets, f)

def test_model():
    model = keras.models.load_model(f'saved_models/model_dim_{20}',custom_objects={"lwlwrap_tf":lwlwrap_tf})
    print("Model Loaded")
    x,y = get_data()
    x = x[..., tf.newaxis].astype("float32")
    x = np.repeat(x,[3],axis=-1)
    # print(x.shape)
    with open('saved_datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)
        print(f"Dataset Keys : {datasets.keys()}")
        train,test = datasets[0]
    x_train,y_train,x_val,y_val = x[train],y[train],x[test],y[test]
    y_pred = model.predict(x_val)
    # multilabel_confusion_matrix(y_val, y_pred)
    print(y_pred.shape,y_val.shape)
    plot_confusion_matrix(y_val,y_pred,data="Val")
    y_pred = model.predict(x_train)
    plot_confusion_matrix(y_train,y_pred,data="Train")

def plot_confusion_matrix(y_true,y_pred,data="Val",n_classes=80):
    df1 = pd.read_csv("./data/sample_submission.csv")
    labels = list(df1.columns)[1:]
    threshold = 0.5
    y_pred = (y_pred > threshold).astype(np.int32)
    # print(y_pred)
    # confusion_matrix = multilabel_confusion_matrix(y_true, y_pred)
    file = open(f'prediction_number_{data}.txt',"w")
    for k in range(4):
        f, axes = plt.subplots(4, 5, figsize=(25, 20))
        axes = axes.ravel()
        # print(axes)
        for i in range(20*k,20*(k+1)):
            disp = ConfusionMatrixDisplay(confusion_matrix(y_true[:, i],
                                                        y_pred[:, i],labels=[0,1]),
                                          display_labels=[0,f"{i}"]
                                        )
            count_1 = np.sum(y_true[:,i])
            count_0 = len(y_true[:,i])-count_1
            line = f"label {labels[i]} : 0 -- {count_0} 1 -- {count_1}\n"
            file.writelines(line)
            disp.plot(ax=axes[i%20])
            disp.ax_.set_title(f'class {labels[i]}')
            if i<10:
                disp.ax_.set_xlabel('')
            if i%5!=0:
                disp.ax_.set_ylabel('')
            disp.im_.colorbar.remove()

        plt.subplots_adjust(wspace=0.10, hspace=0.1)
        f.colorbar(disp.im_, ax=axes)
        # plt.show()
        plt.savefig(f'imgs/confusion_{data}_{k}.png')
    file.close()
    



if __name__ == "__main__":
    # train_model(cnn=True)
    test_model()
    # fusion_train()