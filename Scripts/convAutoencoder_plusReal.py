# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 10:52:03 2019

@author: Karips
"""
import datetime
import os
import pickle
import scipy
import seaborn as sns
from diana_methods import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, UpSampling1D, Reshape
#from keras.layers import advanced_activations
#from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from random import random, seed

seed(50)

def test(df,times):
    df = df.transpose()
    df.hist(column = [i+(12*times) for i in range(12)],bins = 30)
    times+=1
    return times
#os.chdir("C:\\Users\\dimkary\\Box Sync\\SHM Thesis\\Code")
os.chdir("C:\\Users\\dimkary\\Desktop\\Thesis\\Code")
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1" #model will be trained on GPU 1

def min_max_PandasNorm(df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaler = min_max_scaler.fit(x)
    x_norm = scaler.transform(x)
    return pd.DataFrame(x_norm), scaler

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
#        filepath='Reinf.tb',
#        saveResults='EXX_bar', numberOfLongBars=1)

#load data
data = pickle.load( open( "EXX_bar.txt", "rb" ) )
data[0] = np.insert(data[0],0,np.ones((1,np.shape(data[0])[1]))*1e-9,axis = 0)
data[0]*=1000000
scaler = data[0].max().max()
data[0]/= scaler
#
#Dataframes make manipulation easier
df = [pd.DataFrame(i) for i in data]

#Normalize data
#----------min-max normalization---------------#
minmax=[min_max_PandasNorm(d) for d in df]
df_minmax = [i[0] for i in minmax]
#Variable that holds initial preprocessor
preproc=[i[1] for i in minmax]
#----------mean normalization------------------#
df_stand=[(d-d.mean())/d.std() for d in df]
#----------median normalization------------------#
#df_med=[(d-d.median())/(abs(d- d.median())).median() for d in df]
#----------choose preprocessing----------------#
df_norm = df
#df_norm[0]+=max(abs(min(df_norm[0].min())),max(df_norm[0].max())) # JUST FOR STANDARIZATION

times_zero_import = 1
times_firstRow_copied = 100
for i in range(times_firstRow_copied-1):
    df_norm = [pd.concat([df_norm[i][:1],df_norm[i]], ignore_index=True) for i in range(len(df))]
for k in range(times_zero_import):
    df = [pd.concat([df[i][:times_firstRow_copied],df[i]], ignore_index=True) for i in range(len(df))]
    df_norm = [pd.concat([df_norm[i][:times_firstRow_copied],df_norm[i]], ignore_index=True) for i in range(len(df_norm))]

#Add noise to the data:
#noise = [np.random.beta(2,2,df.shape) for df in df_norm]
noise = [np.random.beta(1.2,10,df.shape)*df.mean().mean() for df in df_norm]
df_norm = [df_norm[i]+noise[i] for i in range(len(df_norm))]


plot_scaler = 0.005

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


failure_timestep = 700;
labels = [0 if i<failure_timestep else 1 for i in range(df_norm[0].shape[0])]
[i.insert(0,'Anomaly',labels) for i in df_norm];


correlation_matrix = [df_norm[i].corr() for i in range(1)]
fig, ax = plt.subplots(1,1)
fig.suptitle('Correlation matrix',fontsize=20, fontweight='bold',y=0.9)
sns.heatmap(correlation_matrix[0],square = True, ax=ax,cbar=True,
            cbar_kws={"shrink": .67})
ax.set_title("Bar 1",fontweight='bold')
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
#    act_func = 'relu'
##    act_func = advanced_activations.ReLU(threshold=-0.1, negative_slope=0.5)
#    input_dim = x_train[i].shape[1]
#    encoding_dim = 93
#    input_layer = Input(shape=(input_dim, ))
#    encoder = Dense(encoding_dim, activation=act_func,
#                    activity_regularizer=regularizers.l2(2e-4))(input_layer)
#    encoder = Dense(int(encoding_dim / 2), activation=act_func)(encoder)
#    encoder = Dense(int(encoding_dim / 4), activation=act_func)(encoder)
#    encoder = Dense(int(encoding_dim / 8), activation=act_func)(encoder)
##    encoder = Dense(int(encoding_dim / 15), activation="elu")(encoder)
#    encoder = Dense(int(4), activation=act_func)(encoder)
#    decoder = Dense(int(encoding_dim/ 2), activation=act_func)(encoder)
#    decoder = Dense(int(encoding_dim/ 4), activation=act_func)(decoder)
#    decoder = Dense(int(encoding_dim/ 8), activation=act_func)(decoder)
##    decoder = Dense(int(encoding_dim / 15), activation="elu")(decoder)
#    decoder = Dense(int(encoding_dim), activation=act_func)(decoder)
#    decoder = Dense(input_dim, activation=act_func)(decoder)
    filters =1
    input_sig = Input(shape=(93,1))
    x = Conv1D(128,filters, activation='relu', padding='valid')(input_sig)
    x1 = MaxPooling1D(10)(x)
    x2 = Conv1D(32,filters, activation='relu', padding='valid')(x1)
    x3 = MaxPooling1D(5)(x2)
    flat = Flatten()(x3)
    encoded = Dense(32,activation = 'relu')(flat)
     
#    print("shape of encoded {}".format(K.int_shape(encoded)))
     
    # DECODER 
    x2_ = Conv1D(32, filters, activation='relu', padding='valid')(x3)
    x1_ = UpSampling1D(5)(x2_)
    x_ = Conv1D(128, filters, activation='relu', padding='valid')(x1_)
    upsamp = UpSampling1D(10)(x_)
    flat = Flatten()(upsamp)
    decoded = Dense(93,activation = 'relu')(flat)
    decoded = Reshape((93,1))(decoded)
     
#    print("shape of decoded {}".format(K.int_shape(decoded)))
     
    autoencoder.append(Model(input_sig, decoded))
#    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#    autoencoder.append (Model(inputs=input_layer, outputs=decoder))
    autoencoder[-1].summary()

# 3. Training
    nb_epoch = 30
    batch_size = 20
    autoencoder[-1].compile(optimizer='rmsprop', loss='mse' )
    trainer = np.reshape(x_train[i],(x_train[i].shape[0],x_train[i].shape[1],1))
    history = autoencoder[-1].fit(trainer, trainer,
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
    tester = np.reshape(x_test[k],(x_test[k].shape[0],x_test[k].shape[1],1))
    predictions = autoencoder[k].predict(tester)
    mse = np.mean(np.power(tester - predictions, 2), axis=1)
    mse = np.reshape(mse,(mse.shape[0],))
    errors.append(pd.DataFrame({'reconstruction_error': mse,
                                'Label': y_test[k]}, index=y_test[k].index))
    errors[-1].describe()

    # Testing with train reconstruction error
    predictions_train = autoencoder[k].predict(trainer)
    mse_train = np.mean(np.power(trainer - predictions_train, 2), axis=1)
    mse_train = np.reshape(mse_train,(mse_train.shape[0],))
    errors_train.append(pd.DataFrame({'reconstruction_error': mse_train,
                                      'Label': [1 for i in range(len(mse_train))]}))
    rec_error_threshold_LEVEL0 = (errors_train[k].max()+1*errors_train[k].std())[0]
#    rec_error_threshold_LEVEL1 = 4*rec_error_threshold_LEVEL0
#    rec_error_threshold_LEVEL2 = 8*rec_error_threshold_LEVEL0
    rec_error_threshold_LEVEL1 = (errors[k].max()-2*errors[k].std())[0]
    rec_error_threshold_LEVEL2 = (errors[k].max()-1.25*errors[k].std())[0] 
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
    
    anomalies = errors[-1].index[errors[-1].reconstruction_error >
                       rec_error_threshold_LEVEL0].tolist()
    
#6. All testing
    all_tester = df_norm[k].drop('Anomaly',axis = 1).values
    all_tester = tester = np.reshape(
            all_tester,(all_tester.shape[0],all_tester.shape[1],1))
    all_predictions = autoencoder[k].predict(
            all_tester)
    
    mse_all = np.mean(np.power(all_tester - all_predictions, 2), axis=1)
    mse_all = np.reshape(mse_all,(mse_all.shape[0],))
    all_errors.append(pd.DataFrame({'reconstruction_error': mse_all,
                                 'Label': labels},
    index=np.linspace(1,len(mse_all),len(mse_all),dtype=int)))
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
    ax[1].axhline(rec_error_threshold_LEVEL0, color='blue',
      lw=2.5,linestyle='--',
      label='Error threshold (small damage)')
    ax[1].axhline(rec_error_threshold_LEVEL1, color='#e67300',
      lw=2.5,linestyle='--',
      label='Error threshold (significant damage)')
    ax[1].axhline(rec_error_threshold_LEVEL2, color='red',
      lw=2.5,linestyle='--',
      label='Error threshold (hazardous damage)')
#    ax[1].grid(linestyle = '-', linewidth=0.4)
    ax[1].legend()


####################################################
########### TEST WITH FIRST REAL DATA ##############
####################################################
######### FIRST HERE TO TEST SAVED MODEL ###########
####################################################
    
mat = []
for i in range(1): #Load just the first one to test
    mat.append(scipy.io.loadmat(
            'C:\\Users\\dimkary\\Desktop\\Short_beams\\Exported data\\Real Filtered Data_{0}.mat'.format(i+6))['values'])
with open('Real_EXX.txt', 'wb') as f:
    pickle.dump(mat, f)
    



final_size = df[0].shape[1];
moder = mat[0].shape[1]%final_size 
remainder = mat[0].shape[1]-moder;

mat[0] = mat[0][:,:remainder];
div = mat[0].shape[1]//final_size
shrinked = [];

for i in range(mat[0].shape[0]):
    temp = []
    for j in range(final_size):
        temp.append(np.mean(mat[0][i,div*(j)+1:div*(j+1)]));
    shrinked.append(temp)
    
#######BEAM OUTLIER REMOVAL########
#shrinked = shrinked[:-828] #BEAM1
    
#shrinked = shrinked[:7600]+shrinked[7603:7793]+shrinked[7796:] #BEAM2 
#shrinked = shrinked[:-400] #BEAM2  (Problematic)
    
#shrinked = shrinked[:-400] #BEAM3
#del shrinked[100:120] #BEAM3
    
#shrinked = shrinked[:-105] #BEAM4

#shrinked = shrinked[:-300] #BEAM5    

shrinked = shrinked[:10130]+shrinked[12000:-400] #BEAM6 (Problematic)
###################################

#mat[0] = pd.DataFrame(preproc[0].fit_transform(np.asarray(shrinked)));
mat[0] = np.asarray(shrinked)/scaler

for i in range(len(mat)):
    fig, axs = plt.subplots(1)
    im = axs.imshow(mat[i],aspect=0.02*plot_scaler*max(mat[i].shape)
    /min(mat[i].shape),origin='upper');
    fig.colorbar(im)
    axs.set_xlabel('Element Number')
    axs.set_ylabel('Time step')
    axs.set_title('Bar number %d'%(i+1),fontdict =
              {'fontsize': 15, 'fontweight' : 'bold'})
    plt.show()

errors_real = []
for k in range(len(mat)): 
    tester = np.reshape(mat[k],(mat[k].shape[0],mat[k].shape[1],1))
    predictions_real = autoencoder[k].predict(tester)
    mse_real = np.mean(np.power(tester - predictions_real, 2), axis=1)
    mse_real = np.reshape(mse_real,(mse_real.shape[0],))
    errors_real.append(pd.DataFrame({'reconstruction_error': mse_real,
                                }))
    errors_real[-1].describe()
    plt.figure()
    plt.plot([i+1 for i in list(range(errors_real[-1].shape[0]))],
              errors_real[-1]['reconstruction_error'],'bo',alpha = 0.6)
    plt.title("Reconstruction error (test_data)")
     
    anomalies_real = errors_real[-1].index[errors_real[-1].reconstruction_error
                                 > rec_error_threshold_LEVEL0].tolist()
        
    temp_x = [i+1 for i in list(range(errors_real[-1].shape[0]))]
    temp_y =errors_real[-1]['reconstruction_error'].values

    fig, ax = plt.subplots(2, sharex=True)
    fig.suptitle('Reconstruction Error',fontsize=13, fontweight='bold',y=0.93)
    scat1 = ax[0].scatter(temp_x,
              temp_y,
              alpha = 0.3, marker=".",s=12,
                color = 'palevioletred',label='Experimental data')
    line = ax[0].axhline(rec_error_threshold_LEVEL0, color='blue', lw=2.5,
             linestyle='--', label='Error threshold (small damage)')
    line = ax[0].axhline(rec_error_threshold_LEVEL1, color='orange', lw=2.5,
             linestyle='--', label='Error threshold (significant damage)')
    line = ax[0].axhline(rec_error_threshold_LEVEL2, color='red', lw=2.5,
             linestyle='--', label='Error threshold (hazardous damage)')
    ax[0].grid(linestyle = '-', linewidth=0.4)
    ax[0].legend()
    
    ax[1].plot(temp_x,temp_y)
    ax[1].axhline(rec_error_threshold_LEVEL0, color='blue',
      lw=2.5,linestyle='--',
      label='Error threshold (small damage)')
    ax[1].axhline(rec_error_threshold_LEVEL1, color='#e67300',
      lw=2.5,linestyle='--',
      label='Error threshold (significant damage)')
    ax[1].axhline(rec_error_threshold_LEVEL2, color='red',
      lw=2.5,linestyle='--',
      label='Error threshold (hazardous damage)')
#    ax[1].grid(linestyle = '-', linewidth=0.4)
    ax[1].legend()
    