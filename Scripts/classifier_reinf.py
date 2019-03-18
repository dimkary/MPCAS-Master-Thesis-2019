# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:52:03 2019

@author: Karips
"""
import datetime
import os
import pickle
import seaborn as sns
from diana_methods import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Activation

from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras import regularizers
#from ann_visualizer.visualize import ann_viz
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from random import random
from sklearn.metrics import roc_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE


def test(df,times):
    df = df.transpose()
    df.hist(column = [i+(12*times) for i in range(12)],bins = 30)
    times+=1
    return times

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1

def min_max_PandasNorm(df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    x_norm = min_max_scaler.fit_transform(x)
    return pd.DataFrame(x_norm)    

def train_validation_loss(df_history):
    '''
	 ---------------------------------------------------
	   Parameters: 
           df_history: pandas.Datafram -> Dataframe which
               contains the train and val loss.
               NOTE: first column -> val_loss
                     second column -> train_loss
     --------------------------------------------------
    '''

    train_hist = list(df_history['loss'])
    val_hist = list(df_history['val_loss'])
    #sorry for the map/lambda show off :-(
    steps = list(map(lambda x : x+1, [i for i in range(df_history.shape[0])]))
    
    fig, ax = plt.subplots()
    ax.plot(steps, train_hist, label='Training loss')
    ax.plot(steps, val_hist, label='Validation loss')
    ax.legend(loc='best')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 13
    plt.title('Train and Validation Loss over Epochs',fontdict =
              {'fontsize': 15, 'fontweight' : 'bold'})
    plt.show()  
    return None

#data = Diana3D_EXX_preparation(
#        filepath='C:\\Users\\dimkary\\Desktop\\Temp\\Reinforcement.tb',
#        saveResults='EXX_bar', numberOfLongBars=2)

#load data
data = pickle.load( open( "EXX_bar.txt", "rb" ) )

#Dataframes make manipulation easier
df = [pd.DataFrame(i) for i in data]
times_zero_import = 5
for k in range(times_zero_import):
    df = [pd.concat([df[i][:200]*6,df[i]], ignore_index=True) 
    for i in range(len(df))]
#Normalize data
#----------min-max normalization---------------#
df_minmax=[min_max_PandasNorm(d) for d in df]

#----------mean normalization------------------#
df_stand=[(d-d.mean())/d.std() for d in df]

#----------choose preprocessing----------------#
df_norm = df_stand
#Add noise to the data:
noise = [np.random.normal(0,0.3,df.shape) for df in df_norm]
df_norm = [df_norm[i]+noise[i] for i in range(len(df_norm))]

times = 0
valids =[]

plot_scaler = 0.002

for i in range(len(df_norm)):
    fig, axs = plt.subplots(1)
    im = axs.imshow(df_norm[i],aspect=plot_scaler*max(df_norm[i].shape)
    /min(df_norm[i].shape),origin='upper');
    fig.colorbar(im)
    axs.set_xlabel('Element Number')
    axs.set_ylabel('Time step')
    axs.set_title('Bar number %d'%(i+1),fontdict =
              {'fontsize': 15, 'fontweight' : 'bold'})
    plt.show()
    
# TO EXPLORE THE DATA AND DECIDE THE BEST PREPROCESSING, WE SUM ALL THE
# ELEMENTS FOR ALL TIMESTEPS. THE ONE WITH BIGGER VARIANCE WILL BE CHOSEN
#plt.figure()
#bins = 50
#temp = df[1].sum(axis=1)
#plt.hist(temp,bins = bins,alpha = 0.5)
#temp = df[0].sum(axis=1)
#plt.hist(temp,bins = bins,alpha = 0.5)
#plt.title("No preprocessing")
#
#plt.figure()  
#bins = 50
#temp = df_norm[1].sum(axis=1)
#plt.hist(temp,bins = bins,alpha = 0.5)
#temp = df_norm[0].sum(axis=1)
#plt.hist(temp,bins = bins,alpha = 0.5)    
#plt.title("Mean normalization")
# 
#plt.figure()  
#bins = 50
#temp = df_minmax[1].sum(axis=1)
#plt.hist(temp,bins = bins,alpha = 0.5)
#temp = df_minmax[0].sum(axis=1)
#plt.hist(temp,bins = bins,alpha = 0.5)
#plt.title("Min-max normalization")


#Create synthetic labels for dataset (somewhat arbitrarily) 
#and add it to the dataframe

failure_timestep = 2000;
labels = [0 if i<failure_timestep else 1 for i in range(df_norm[0].shape[0])]
[i.insert(0,'Anomaly',labels) for i in df_norm];


correlation_matrix = [df_norm[i].corr() for i in range(2)]
fig, ax = plt.subplots(1,2)
fig.suptitle('Correlation matrix',fontsize=20, fontweight='bold',y=0.9)
sns.heatmap(correlation_matrix[0],square = True,
            ax=ax[0],cbar=True,cbar_kws={"shrink": .67})
sns.heatmap(correlation_matrix[1],square = True,
            ax=ax[1],cbar=True,cbar_kws={"shrink": .67})
ax[0].set_title("Bar 1",fontweight='bold')
ax[1].set_title("Bar 2",fontweight='bold')
##--------------DATA DIVISION----------------#
## 1. Preallocate X_train/test as "train_test_split" objects
x_train, x_test , y_test , y_train= [],[],[],[]

for i in range(len(data)):
    temp_train, temp_test = train_test_split(df_norm[i],test_size = 0.2,
                                   random_state = 10)
    temp_prediction = temp_train['Anomaly']
    sm = SMOTE(random_state=12, ratio = 1.0)
    temp_train, temp_prediction = sm.fit_sample(temp_train.drop(['Anomaly'],axis=1), temp_prediction)
    x_train.append(temp_train); 
    x_test.append(temp_test.drop(['Anomaly'],axis=1));
    y_train.append(temp_prediction);
    y_test.append(temp_test['Anomaly']);
    print(f'Training data size for bar {i}  :', x_train[i].shape)
    print(f'Validation data size for bar {i} :', x_test[i].shape)
    
    
#--------------TRAINING----------------#
# 2. Create network with "keras" dense layers
# No of Neurons in each Layer [9,6,3,2,3,6,9]
classifier, histories, errors, all_errors, all_anomalies = [],[],[],[],[]
stop_callback =EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=15,
                              verbose=0, mode='auto')
t_ini = datetime.datetime.now()

for i in range(len(x_train)): 
    input_dim = x_train[i].shape[1]
    model = Sequential()
    model.add(Dense(32, input_dim=input_dim))
    model.add(Activation('elu'))
    model.add(Dense(16, input_dim=input_dim))
    model.add(Activation('elu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

    classifier.append (model)
    classifier[-1].summary()

# 3. Training
    nb_epoch = 200
    batch_size = 100
  
    history = classifier[-1].fit(x_train[i], y_train[i],
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_split=0.2,
                            verbose=1, callbacks = [stop_callback])
    
    histories.append(pd.DataFrame(history.history))
    
t_fin = datetime.datetime.now()
print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))
##Plot losses
[train_validation_loss(his) for his in histories]
#
#--------------TESTING----------------#
# 4. Testing
# use MAX_ERROR from both bars to determine the final threshold
losses, accuracies,predictions = [],[],[]
for k in range(len(x_train)):    
    loss, acc = classifier[k].evaluate(x_test[k], y_test[k])
    losses.append(loss); accuracies.append(accuracies)
    predictions.append(classifier[k].predict(x_test[k]))
    
#
#7. ROC curve, AUC
for i in range(len(df_norm)):
    plt.figure()
    fpr, tpr, thresholds = roc_curve(y_test[i], predictions[i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.ylim([0.0, 1])
    plt.xlim([-0.02, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic, Bar %d' % (i+1),
              fontdict={'fontsize': 15, 'fontweight' : 'bold'})
    plt.legend(loc="lower right")
    
#8. Confusion matrix
for i in range(len(predictions)):
    #Casted to 'int' cause of float errors (check values in predictions)
    conf_mat = confusion_matrix(y_test[i], [ int(j) for j in predictions[i]])
    fig, ax = plt.subplots(2, sharex=True)
    fig.suptitle('Confusion Matrix for Bar %d' % (i+1),
                 fontsize=13, fontweight='bold',x=0.435)
    sns.heatmap(conf_mat, annot=True,ax=ax[0],fmt="d" );
    sns.heatmap(conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis],
                annot=True,ax=ax[1] )
    ax[0].set_title("Unormalized",fontweight='bold')
    ax[1].set_title("Normalized",fontweight='bold')

