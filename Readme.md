# Readme for Assessment 2 FreeAudio Classification

1. data_analysis.py creates the features for the audio data. 
2. To preprocess the features further, I trim the audio to remove sound below 60 db and instead of padding      after extracting features, I am padding the audio signal first and then extractng the relevant features.
3. Using librosa I extract 20 dimension MFCC features, 30 dimension mel spectrogram features, 12 dimension chroma and 6 dimension tonnetz features.
4. In model.py I tried different types of models. I tried to train an LSTM and CNN model from scratch. They however did not generate a good enough lwlwrap even on the full training data.
5. I tried a fusion model as well that processes each set of features dfferently and then concatenates them in the end and does the classification.
6. Finally, I tried a pretrained vgg19 model and fine-tuned it on our train_curated data. This led to an lwlwrap of 0.98 on the train data and 0.47 on the validation data.
7. The model was trained for 15 epochs. And the loss was still decreasing for both train and val. Training further might increase the performance howver, each epoch takes almost an hour.
8. I have also created the confusion matrices for each label considering binary classification of each class.
9. In saved_datasets.pkl I save the indices that correspond to the validation and training data.
10. freeaudio_features_trimmed.npy contains the features of the train_curated data.
11. imgs/ contains the confusion matrix images
12. saved_models contains the saved tensorflow models.
13. data_analysis.ipynb contains plots corresponding to the different features of the data and also contains some EDA on the train_curated dataset.
14. prediction_number.txt contains the number of positive and negative samples in the train and validation datasets.
