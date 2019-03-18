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
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
#from ann_visualizer.visualize import ann_viz
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from random import random

def test(df,times):
    df = df.transpose()
    df.hist(column = [i+(12*times) for i in range(12)],bins = 30)
    times+=1
    return times
os.chdir("C:\\Users\\dimkary\\Box Sync\\SHM Thesis\\Code")
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
df = [d[199:] for d in df]
#Normalize data
#----------min-max normalization---------------#
df_minmax=[min_max_PandasNorm(d) for d in df]

#----------mean normalization------------------#
df_stand=[(d-d.mean())/d.std() for d in df]
#----------median normalization------------------#
df_med=[(d-d.median())/(abs(d- d.median())).median() for d in df]
#----------choose preprocessing----------------#
df_norm = df_stand

times_zero_import = 4
df = [pd.DataFrame(i) for i in data]
for i in range(199):
    df_norm = [pd.concat([df_norm[i][:1],df_norm[i]], ignore_index=True) for i in range(len(df))]
for k in range(times_zero_import):
    df = [pd.concat([df[i][:200],df[i]], ignore_index=True) for i in range(len(df))]
    df_norm = [pd.concat([df_norm[i][:200],df_norm[i]], ignore_index=True) for i in range(len(df_norm))]

#Add noise to the data:
noise = [np.random.normal(0,0.3,df.shape) for df in df_norm]
df_norm = [df_norm[i]+noise[i] for i in range(len(df_norm))]


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


failure_timestep = 1800;
labels = [0 if i<failure_timestep else 1 for i in range(df_norm[0].shape[0])]
[i.insert(0,'Anomaly',labels) for i in df_norm];


correlation_matrix = [df_norm[i].corr() for i in range(2)]
fig, ax = plt.subplots(1,2)
fig.suptitle('Correlation matrix',fontsize=20, fontweight='bold',y=0.9)
sns.heatmap(correlation_matrix[0],square = True, ax=ax[0],cbar=True,
            cbar_kws={"shrink": .67})
sns.heatmap(correlation_matrix[1],square = True, ax=ax[1],cbar=True,
            cbar_kws={"shrink": .67})
ax[0].set_title("Bar 1",fontweight='bold')
ax[1].set_title("Bar 2",fontweight='bold')
##--------------DATA DIVISION----------------#
## 1. Preallocate X_train/test as "train_test_split" objects
x_train, x_test , y_test = [],[],[]
for i in range(len(data)):
    temp_train, temp_test = train_test_split(df_norm[i],test_size = 0.2,
                                   random_state = 85)
    x_train.append(temp_train); x_test.append(temp_test);

for i in range(len(x_train)):
    x_train[i] = x_train[i][x_train[i]['Anomaly'] == 0]
    x_train[i] = x_train[i].drop(['Anomaly'], axis=1)
    y_test.append(x_test[i]['Anomaly'])
    x_test[i]  = x_test[i].drop(['Anomaly'], axis=1)
    x_train[i] = x_train[i].values
    x_test[i]  = x_test[i].values
    print(f'Training data size for bar {i}  :', x_train[i].shape)
    print(f'Validation data size for bar {i} :', x_test[i].shape)
    
#--------------TRAINING----------------#
# 2. Create network with "keras" dense layers
# No of Neurons in each Layer [9,6,3,2,3,6,9]
autoencoder, histories, errors, all_errors, all_anomalies = [],[],[],[],[]

t_ini = datetime.datetime.now()
for i in range(len(x_train)): 
    input_dim = x_train[i].shape[1]
    encoding_dim = 200
    input_layer = Input(shape=(input_dim, ))
    encoder = Dense(encoding_dim, activation="elu",
                    activity_regularizer=regularizers.l1(10e-3))(input_layer)
    encoder = Dense(int(encoding_dim / 2), activation="elu")(encoder)
    encoder = Dense(int(encoding_dim / 2), activation="elu")(encoder)
    encoder = Dense(int(2), activation="elu")(encoder)
    decoder = Dense(int(encoding_dim/ 2), activation='elu')(encoder)
    decoder = Dense(int(encoding_dim/ 2), activation='elu')(encoder)
    decoder = Dense(int(encoding_dim), activation='elu')(decoder)
    decoder = Dense(input_dim, activation='elu')(decoder)
    autoencoder.append (Model(inputs=input_layer, outputs=decoder))
    autoencoder[-1].summary()

# 3. Training
    nb_epoch = 200
    batch_size = 50
    autoencoder[-1].compile(optimizer='rmsprop', loss='mse' )
    
    history = autoencoder[-1].fit(x_train[i], x_train[i],
                            epochs=nb_epoch,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_split=0.2,
                            verbose=1
                            )
    

    
    
    histories.append(pd.DataFrame(history.history))
    
t_fin = datetime.datetime.now()
print('Time to run the model: {} Sec.'.format((t_fin - t_ini).total_seconds()))
##Plot losses
[train_validation_loss(his) for his in histories]
#
#--------------TESTING----------------#
# 4. Testing
# use MAX_ERROR from both bars to determine the final threshold
errors_train = []
for k in range(len(x_train)): 
    predictions = autoencoder[k].predict(x_test[k])
    mse = np.mean(np.power(x_test[k] - predictions, 2), axis=1)
    errors.append(pd.DataFrame({'reconstruction_error': mse,
                                'Label': y_test[k]}, index=y_test[k].index))
    errors[-1].describe()
    rec_error_threshold_LEVEL1 = np.floor(errors[k].max()-errors[k].std())[0]
    rec_error_threshold_LEVEL2 = np.floor(errors[k].max()+errors[k].std())[0]   
    # Testing with train reconstruction error
    predictions_train = autoencoder[k].predict(x_train[k])
    mse_train = np.mean(np.power(x_train[k] - predictions_train, 2), axis=1)
    errors_train.append(pd.DataFrame({'reconstruction_error': mse_train,
                                      'Label': [1 for i in range(len(mse_train))]}))
    rec_error_threshold_LEVEL0 = np.ceil(errors_train[k].max())[0]
#    rec_error_threshold = np.floor(errors[k].max()-2*errors[k].std())[0]
#5. Plot reconstruction error
    plt.figure()
    plt.plot([i+1 for i in list(range(errors[-1].shape[0]))],
              errors[-1]['reconstruction_error'],'bo',alpha = 0.6)
    plt.title("Reconstruction error (test_data)")
    
    plt.figure()
    plt.plot([i+1 for i in list(range(errors_train[-1].shape[0]))],
              errors_train[-1]['reconstruction_error'],'bo',alpha = 0.6)
    plt.title("Reconstruction error (train_data)")
    
    anomalies = errors[-1].index[errors[-1].reconstruction_error > rec_error_threshold_LEVEL0].tolist()
    
#6. All testing
    all_predictions = autoencoder[k].predict(
            df_norm[k].drop('Anomaly',axis = 1))
    
    mse_all = np.mean(np.power(df_norm[k].drop(
            'Anomaly',axis = 1) - all_predictions, 2), axis=1)
    all_errors.append(pd.DataFrame({'reconstruction_error': mse_all,
                                 'Label': labels}, index=mse_all.index))
    all_anomalies.append(
            all_errors[-1].index[
                    all_errors[-1].reconstruction_error > 
                    np.std(
                    all_errors[-1]['reconstruction_error'].values)].tolist())
    
    
    temp_x = [i+1 for i in list(range(all_errors[-1].shape[0]))]
    temp_y =all_errors[-1]['reconstruction_error'].values
    colors = ['red','green']

    fig, ax = plt.subplots(2, sharex=True)
    fig.suptitle('Reconstruction Error',fontsize=13, fontweight='bold',y=0.93)
    scat1 = ax[0].scatter(temp_x[0:failure_timestep],
              temp_y[0:failure_timestep],
              alpha = 0.4, marker="o",s=12,
                color = 'g',label='Observed')
    scat2 = ax[0].scatter(temp_x[failure_timestep:],
              temp_y[failure_timestep:],
              alpha = 0.4, marker="o",s=12,
                color = 'r',label='Non-observed')   
    line = ax[0].axhline(rec_error_threshold_LEVEL0, color='blue', lw=2.5,
             linestyle='--', label='Error threshold (small damage)')
    line = ax[0].axhline(rec_error_threshold_LEVEL1, color='orange', lw=2.5,
             linestyle='--', label='Error threshold (significant damage)')
    line = ax[0].axhline(rec_error_threshold_LEVEL2, color='red', lw=2.5,
             linestyle='--', label='Error threshold (hazardous damage)')
    ax[0].grid(linestyle = '-', linewidth=0.4)
    ax[0].legend()
    
    ax[1].plot(temp_x,temp_y)
    ax[1].axhline(rec_error_threshold_LEVEL0, color='blue', lw=2.5,linestyle='--',
      label='Error threshold (small damage)')
    ax[1].axhline(rec_error_threshold_LEVEL1, color='#e67300', lw=2.5,linestyle='--',
      label='Error threshold (significant damage)')
    ax[1].axhline(rec_error_threshold_LEVEL2, color='red', lw=2.5,linestyle='--',
      label='Error threshold (hazardous damage)')
#    ax[1].grid(linestyle = '-', linewidth=0.4)
    ax[1].legend()


