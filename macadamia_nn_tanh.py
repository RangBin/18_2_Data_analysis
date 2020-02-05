# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 20:10:53 2018

@author: Seona Lee
# Data Analysis Team project - Team Macadamia
Categorize human Activity using sensor data collected using smartphone
Using two data set for our evaluation:
- Human activity recognition with smartphone
- Simplified human activity recognition
This code build model using Keras on Tensorflow.
Use tanh as activation function
"""
#import modules needed
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix

import itertools
import matplotlib.pyplot as plt
import numpy as np

### plotting confusion matrix by Yeonjin Jin
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

####################    
# download data set from (https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones)
hu_file_root="./human-activity-recognition-with-smartphones/"
# download second data set from (https://www.kaggle.com/mboaglio/simplifiedhuarus)
sp_file_root="./simplifiedhuarus/"

#load first dataset from csv file
hu_train = pd.read_csv(hu_file_root+"train.csv")
hu_test = pd.read_csv(hu_file_root+"test.csv")

#shuffling data
hu_mixed_train = shuffle(hu_train)
hu_mixed_test = shuffle(hu_test)

# Seperate subject information
hu_subject_training_data = hu_mixed_train['subject']
hu_subject_testing_data = hu_mixed_test['subject']

# Seperate labels
hu_training_labels = hu_mixed_train['Activity']
hu_testing_labels = hu_mixed_test['Activity']

# Encode categorical labels into numerical target labels
class_names = ['LAYING', 'SITTING', 'STANDING', 'WALKING', 'WALKING_DOWNSTAIRS',
       'WALKING_UPSTAIRS']
le = preprocessing.LabelEncoder()
le = le.fit(class_names)
enc_training_labels = le.transform(hu_training_labels)
enc_testing_labels = le.transform(hu_testing_labels)

# Drop labels and subject information from data
hu_final_train = hu_mixed_train.drop(['subject', 'Activity'], axis=1)
hu_final_test = hu_mixed_test.drop(['subject', 'Activity'], axis=1)

# Encode targets as one-hot label vectors
oh_training_labels = keras.utils.to_categorical(enc_training_labels)
oh_testing_labels = keras.utils.to_categorical(enc_testing_labels)

### Build a neural network for this classification task
# - First model uses tanh activation function
model_tanh = keras.models.Sequential()
model_tanh.add(keras.layers.Dense(units=96, input_dim = hu_final_train.shape[1], activation = 'tanh'))
model_tanh.add(keras.layers.Dropout(0.5))
model_tanh.add(keras.layers.Dense(units=30, activation = 'tanh'))
model_tanh.add(keras.layers.Dense(units=24, activation = 'tanh'))
model_tanh.add(tf.keras.layers.Dense(6, activation=tf.nn.softmax))

# add early stopping callback and compare results
early_stopping = [tf.keras.callbacks.EarlyStopping(monitor = 'val_acc', min_delta = .002, patience = 7, mode = 'auto')]
model_tanh.compile(optimizer = "adam", loss = 'categorical_crossentropy', metrics = ['accuracy'])
# callbacks = early_stopping,
history_tanh = model_tanh.fit(hu_final_train.values, oh_training_labels, epochs = 120, batch_size = 30, callbacks = early_stopping,
                    verbose = 2, validation_split = .15, shuffle=True)

# print evaluation result - using first set's test dataset
val_loss, val_acc = model_tanh.evaluate(hu_final_test.values,oh_testing_labels)
print("Loss : "+str(val_loss))
print("Accuracy : "+str(val_acc))

# plot - confusion matrix
predictions = model_tanh.predict(hu_final_test)
predict_result = [class_names[np.argmax(w)] for w in predictions]

nn_cfs = confusion_matrix(hu_testing_labels, predict_result)
plot_confusion_matrix(nn_cfs, classes=class_names,
                      title='')
plt.show()

# plot - loss evaluation
plt.plot(history_tanh.history['loss'])
plt.plot(history_tanh.history['val_loss'])
plt.title('categorical cross entropy loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc = 'upper right' )
plt.show()

# plot - accuracy evaluation
plt.plot(history_tanh.history['acc'])
plt.plot(history_tanh.history['val_acc'])
plt.title('Accuracy evaluation')
plt.ylabel('Accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validation accuracy'], loc = 'lower right' )
plt.show()

#########################
# import and preprocess on second dataset
sp_train = pd.read_csv(sp_file_root+"train.csv")
sp_mixed_train = shuffle(sp_train)
sp_training_labels = sp_mixed_train['activity']
sp_enc_training_labels = le.transform(sp_training_labels)
sp_oh_training_labels = keras.utils.to_categorical(sp_enc_training_labels)
sp_final_train = sp_mixed_train.drop(['rn', 'activity'], axis=1)

# evaluate result of our model on second dataset
sp_loss, sp_acc = model_tanh.evaluate(sp_final_train.values, sp_oh_training_labels)
print("++ Evaluation on another set")
print("Loss : "+str(sp_loss))
print("Accuracy : "+str(sp_acc))

# plot confusion matrix on second dataset
sp_predictions = model_tanh.predict(sp_final_train)
sp_predict_result = [class_names[np.argmax(w)] for w in sp_predictions]
sp_nn_cfs = confusion_matrix(sp_training_labels, sp_predict_result)
plot_confusion_matrix(sp_nn_cfs, classes=class_names,
                      title='')

#EOF
