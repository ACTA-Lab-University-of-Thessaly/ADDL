import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from matplotlib import gridspec

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import  Flatten ,LSTM, Dense, RepeatVector,TimeDistributed, Conv1D, MaxPool1D, Add, Concatenate, Input, Dropout, Cropping1D, Conv1DTranspose
from copy import deepcopy as dc
import time
from sklearn.model_selection import KFold
import random
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

# df1 = pd.read_csv('sample_3months.csv', delimiter=';', index_col='ORDER')
# df1 = df1.apply(lambda x: x.str.replace(',','.').astype(float), axis = 1)

# df_specs1 = pd.read_csv('sample_3months_specs.csv', delimiter=';',usecols = ['HAND_PUMP','FLOW_RATE', 'PRESSURE_SWITCH','TANK_TYPE','BLOCK_VALVE','ORDER'], index_col='ORDER')

df1 = pd.read_csv('sample_2_year_data_pt2.csv', delimiter=';', index_col='ORDER')
#df2 = df2.apply(lambda x: x.str.replace(',','.').astype(float), axis = 1)

df2 = pd.read_csv('sample_2_year_data_pt1.csv', delimiter=';', index_col='ORDER')
#df3 = df3.apply(lambda x: x.str.replace(',','.').astype(float), axis = 1)
df2.drop(['Unnamed: 8'],axis=1, inplace=True)

df_specs = pd.read_csv('sample_2_year_specs.csv', delimiter=';',usecols = ['HAND_PUMP','FLOW_RATE', 'PRESSURE_SWITCH','TANK_TYPE','BLOCK_VALVE','ORDER'], index_col='ORDER')
#orders = pd.unique(df.index)
# df3 = df.join(df2, how = 'outer')

df= pd.concat([df1,df2])


dtfrm = df.join(df_specs,how = 'outer')

orders = np.unique(dtfrm.index)
flow_rate = np.unique(dtfrm.FLOW_RATE)
hand_pump = np.unique(dtfrm.HAND_PUMP)
switch = np.unique(dtfrm.PRESSURE_SWITCH)
tanks = np.unique(dtfrm.TANK_TYPE)
valves = np.unique(dtfrm.BLOCK_VALVE)

#dictionairy with normalized encoded values
dic_flow_rate = dict(zip(flow_rate,np.asarray( list(range(len(flow_rate))) )/(len(flow_rate)-1)))
dic_switch = dict(zip(switch,np.asarray( list(range(len(switch))) )/(len(switch)-1)))

# list_dfs = []
# for i in range(len(orders)):
#     list_dfs.append(dtfrm.loc[orders[i]])
#     dtfrm.replace(0,np.nan).loc[orders[i]].plot(x = 'TIME', subplots = True)
#     plt.suptitle('Order {}\npump = {}, flowrate = {}, tanktype = {}\npressureswitch = {}, blockvalve = {}'.format(orders[i],
#  dtfrm['HAND_PUMP'].loc[orders[i]].iloc[0],
# dtfrm['FLOW_RATE'].loc[orders[i]].iloc[0], dtfrm['TANK_TYPE'].loc[orders[i]].iloc[0], dtfrm['PRESSURE_SWITCH'].loc[orders[i]].iloc[0], dtfrm['BLOCK_VALVE'].loc[orders[i]].iloc[0] ), fontsize = 9)
#     plt.savefig('index_{}'.format(i),dpi= 220)


def find_statistics(column):
    mean = []
    maxim = []
    std = []
    for i in orders:
        m = dtfrm.loc[i][column].replace(0,np.nan).mean()
        mx = dtfrm.loc[i][column].replace(0,np.nan).max()
        st = dtfrm.loc[i][column].replace(0,np.nan).std()
        mean.append(m)
        maxim.append(mx)
        std.append(st)
    return mean,maxim,std

columns = ['SPEED_UP','SPEED_DOWN','NOISE_UP','NOISE_DOWN','PRESSURE_UP','PRESSURE_DOWN']

# spup_mean,spup_max,spup_std = find_statistics(columns[0])
# spdn_mean,spdn_max,spdn_std = find_statistics(columns[1])
# noup_mean,noup_max,noup_std = find_statistics(columns[2])
# nodn_mean,nodn_max,nodn_std = find_statistics(columns[3])
# peup_mean,peup_max,peup_std = find_statistics(columns[4])
# pedn_mean,pedn_max,pedn_std = find_statistics(columns[5])

# dtfrm2 = pd.DataFrame({'SPEED_UP_MEAN':spup_mean,
# 'SPEED_UP_MAX':spup_max,
# 'SPEED_UP_STD':spup_std,
# 'SPEED_DOWN_MEAN':spdn_mean,
# 'SPEED_DOWN_MAX':spdn_max,
# 'SPEED_DOWN_STD':spdn_std,
# 'NOISE_UP_MEAN':noup_mean,
# 'NOISE_UP_MAX':noup_max,
# 'NOISE_UP_STD':noup_std,
# 'NOISE_DOWN_MEAN':nodn_mean,
# 'NOISE_DOWN_MAX':nodn_max,
# 'NOISE_DOWN_STD':nodn_std,
# 'PRESSURE_UP_MEAN':peup_mean,
# 'PRESSURE_UP_MAX':peup_max,
# 'PRESSURE_UP_STD':peup_std,
# 'PRESSURE_DOWN_MEAN':peup_mean,
# 'PRESSURE_DOWN_MAX':peup_max,
# 'PRESSURE_DOWN_STD':peup_std}, index = orders)


def to_numerical(column, uniques):
    values = []
    for i in orders:
        value = dtfrm.loc[i][column].iloc[0]
        for i in range(len(uniques)):
            if value == uniques[i]:
                values.append(i)
    return values


def to_type(column, uniques):
    values = []
    for i in orders:
        value = dtfrm.loc[i][column].iloc[0]
        for i in range(len(uniques)):
            if value == uniques[i]:
                values.append(uniques[i])
    return values


# kmeans_flow_rate = KMeans(len(flow_rate)).fit(dtfrm2)
# kmeans_pumps = KMeans(len(hand_pump)).fit(dtfrm2)
# kmeans_switch = KMeans(len(switch)).fit(dtfrm2)
# kmeans_tanks = KMeans(len(tanks)).fit(dtfrm2)
# kmeans_valves = KMeans(len(valves)).fit(dtfrm2)



# dtfrm2['HAND_PUMP'] = to_numerical('HAND_PUMP',hand_pump)
# dtfrm2['FLOW_RATE'] = to_numerical('FLOW_RATE',flow_rate)
# dtfrm2['PRESSURE_SWITCH'] = to_numerical('PRESSURE_SWITCH',switch)
# dtfrm2['TANK_TYPE'] = to_numerical('TANK_TYPE',tanks)
# dtfrm2['BLOCK_VALVE'] = to_numerical('BLOCK_VALVE',valves)

# dtfrm2['FLOW_RATE_TYPE'] = to_type('FLOW_RATE',flow_rate)
# dtfrm2['HAND_PUMP_TYPE'] = to_type('HAND_PUMP',hand_pump)
# dtfrm2['PRESSURE_SWITCH_TYPE'] = to_type('PRESSURE_SWITCH',switch)
# dtfrm2['TANK_TYPE_TYPE'] = to_type('TANK_TYPE',tanks)
# dtfrm2['BLOCK_VALVE_TYPE'] = to_type('BLOCK_VALVE',valves)



def plot_hist_threshold(threshold = 0.4, column = 'SPEED_UP_MAX', types = 'FLOW_RATE_TYPE'):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].hist(dtfrm2.loc[dtfrm2[column]<= threshold][types], color = 'g')
    axs[0].set_title('Histogram of {} values\nBelow {}'.format(column,threshold))
    a = np.unique(dtfrm2.loc[dtfrm2[column] <= threshold][types])
    axs[1].hist(dtfrm2.loc[dtfrm2[column] > threshold][types])
    b=np.unique(dtfrm2.loc[dtfrm2[column]> threshold][types]) 
    
    plt.show()
    print(a, b)
    print('items below threshold {}, items above threshold = {}'.format(len(a),len(b)))
    return a,b


def slice_dfs(threshold = 201):
    
    for n,i in enumerate(orders):
        ln = len(dtfrm.loc[i])
        if n == 0:
            if ln > threshold:
                df_cropped = dtfrm.loc[i].iloc[:threshold]
            else:
                df_cropped = dtfrm.loc[i]
        else:
            if ln > threshold:
                df_cropped = pd.concat([df_cropped,dtfrm.loc[i].iloc[:threshold]])
            else:
                df_cropped = pd.concat([df_cropped,dtfrm.loc[i]])
    return df_cropped


def create_sequences(df):
    encoded_vectors = []
    for n,i in enumerate(orders):
        if n == 0:
            up = df.loc[i][['SPEED_UP','NOISE_UP','PRESSURE_UP']].to_numpy()
            up = up.reshape((1,up.shape[0],up.shape[1]))
            down = df.loc[i][['SPEED_DOWN','NOISE_DOWN','PRESSURE_DOWN']].to_numpy()
            down = down.reshape((1,down.shape[0],down.shape[1]))
            final = np.concatenate((up, down))

            flows = dic_flow_rate[df.loc[i]['FLOW_RATE'].iloc[0]]
            switches = dic_switch[df.loc[i]['PRESSURE_SWITCH'].iloc[0]]
            encoded_vectors.append(list((flows,switches)))
            encoded_vectors.append(list((flows,switches)))
        else:
            up = df.loc[i][['SPEED_UP','NOISE_UP','PRESSURE_UP']].to_numpy()
            up = up.reshape((1,up.shape[0],up.shape[1]))
            down = df.loc[i][['SPEED_DOWN','NOISE_DOWN','PRESSURE_DOWN']].to_numpy()
            down = down.reshape((1,down.shape[0],down.shape[1]))
            final2 = np.concatenate((up, down))
            final = np.concatenate((final, final2))

            flows = dic_flow_rate[df.loc[i]['FLOW_RATE'].iloc[0]]
            switches = dic_switch[df.loc[i]['PRESSURE_SWITCH'].iloc[0]]
            encoded_vectors.append(list((flows,switches))) #up
            encoded_vectors.append(list((flows,switches))) #down


    return final,np.asarray(encoded_vectors)


#[:,:,1] = 91.2, [:,:,2] = 53.98

df_cropped= slice_dfs()
x, vectors = create_sequences(df_cropped)
x[:,:,1] = x[:,:,1]/91.2
x[:,:,2] = x[:,:,2]/53.98
test = x[int(len(x)*0.9):]
x = x[:int(len(x)*0.9)]

dummys_array = np.asarray(np.load('dummy_data.npy'))

model = tf.keras.models.load_model('cnn2.h5')

def to_dt(prediction_data, test_data, save = False):
    noise = 91.2
    pressure = 53.98
    columns = ['Speed','Noise','Pressure']
    index = np.asarray(list(range(201))) * 0.1

    predictions = dc(prediction_data)
    test = dc(test_data)

    predictions[:,1] = predictions[:,1]*91.2
    test[:,1] = test[:,1]*91.2
    predictions[:,2] = predictions[:,2]*53.98
    test[:,2] = test[:,2]*53.98
    predictions[predictions<0.001] = 0
    df_pred = pd.DataFrame(predictions,columns=columns, index = index)
    df_real = pd.DataFrame(test,columns=columns, index = index)
    fig, axes = plt.subplots(figsize=(9,13), nrows=3, ncols=1, sharex=True, tight_layout= True)
    for i in range(len(df_pred.columns)):
        axes[i].plot(df_real[df_real.columns[i]], label ='Actual Data', color = 'b', linewidth = 3)
        axes[i].plot(df_pred[df_pred.columns[i]], label ='Reconstruction', color = 'g')
        axes[i].set_title(df_pred.columns[i])
        axes[i].fill_between(df_pred.index, df_real[df_real.columns[i]],df_pred[df_pred.columns[i]], color = 'gray', label = 'error')
        axes[i].legend(loc = 'upper right')
    if save:
        plt.savefig('reconstruction')
    plt.show()




def new_model():
    input1 = layers.Input(shape = (x.shape[1],x.shape[2]))
    en_lstm1_1 = LSTM(128, return_sequences = True)(input1)
    en_lstm2 = LSTM(128)(en_lstm1_1)
    repeat = RepeatVector(x.shape[1])(en_lstm2)
    conc = layers.Add()([repeat,en_lstm1_1])
    de_lstm1 = LSTM(128, return_sequences = True)(conc)


    out = LSTM(x.shape[2],return_sequences = True)(de_lstm1)
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    model = Model(input1, out)
    model.compile(optimizer = 'adam', loss = 'mse', metrics = metrics)
    return model


def evaluate_model(trx, tst, n_folds = 5):
    scores_mse,scores_mae = list(), list()
    times = []
    histories = []
    models = []
    kfold = KFold(n_folds,shuffle=True)
    

    for train, test in kfold.split(trx):
        model = new_model()
        train_x = trx[train]
        eval_x = trx[test]
        batch = 64
        eval_batch = 32
        callbacks = [tf.keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = True)]
        start = time.time()
        history = model.fit(train_x,train_x,epochs=140,batch_size = batch, validation_batch_size = eval_batch, callbacks=callbacks,
         validation_data= (eval_x,eval_x), verbose = 0)
        stop = time.time()
        s, a = model.evaluate(tst,tst) 
        scores_mse.append(s)
        scores_mae.append(a)
        times.append(stop-start)
        histories.append(history)
        models.append(model)
    return scores_mse,scores_mae,histories,times, models


def boxplots(lis, title = ''):
    fig, axs = plt.subplots(nrows = 1, ncols = 2, tight_layout=True, figsize = (8,5) )
    Labels = ['MSE', 'MAE']
    titles = ['Mean Square Error', 'Mean Absolute Error']
    colors = ['c', 'lime']
    for i in range(len(lis)):

        axs[i].boxplot(lis[i], labels = [Labels[i]], showmeans= True, meanline=True, patch_artist= True, boxprops=dict(facecolor = colors[i]),medianprops=dict(color = 'r'), meanprops=dict(color = 'b'))
        axs[i].set_title(titles[i])
        axs[i].set_ylabel('Error')
        axs[i].annotate('Median = {}\n Mean = {}\n  std = {}'.format(np.round(np.median(lis[i]),5),
             np.round(np.mean(lis[i]),5),
              np.round(np.std(lis[i]),5)),
              xy=(.02,.5),xycoords = 'axes fraction', fontsize = 9)
    plt.suptitle('{}\n5-fold Cross Validation'.format(title),x = 0.53, y = 0.95)
    plt.savefig('{}_boxplot.png'.format(title[:4]), dpi = 300)

def new_model_cnn():
    input1 = Input(shape = (x.shape[1],x.shape[2]))
    en_lstm1_11 = Conv1D(64, 15, strides = 3, padding = 'same',activation='relu')(input1)
    en_lstm1_1 = Conv1D(32, 9, strides = 2, padding = 'same',activation='relu')(en_lstm1_11)
    en_lstm1_1 = Conv1D(16, 7, strides = 2, padding = 'same',activation='relu')(en_lstm1_1)
    en_lstm1_1 = Conv1D(8, 3, strides = 1, padding = 'same',activation='relu')(en_lstm1_1)
    pool = MaxPool1D()(en_lstm1_1)
    de_lstm1_1 = Conv1DTranspose(16,3, strides = 2, padding = 'same',activation='relu')(pool)
    de_lstm1_1 = Conv1DTranspose(32,7, strides = 2, padding = 'same',activation='relu')(de_lstm1_1)
    de_lstm1_1 = Conv1DTranspose(64,9, strides = 2, padding = 'same',activation='relu')(de_lstm1_1)

    crop = layers.ZeroPadding1D((2,1))(de_lstm1_1)
    #conc = layers.Add()([crop,en_lstm1_11])
    de_lstm1_1 = Conv1DTranspose(64,15, strides = 3, padding = 'same',activation='relu')(crop)
    #crop = Cropping1D(cropping = (0,1))(de_lstm1_1)

    out = Conv1D(x.shape[2],1,padding = 'same')(de_lstm1_1)
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    model = Model(input1, out)
    model.compile(optimizer = 'adam', loss = 'mse', metrics = metrics)
    return model



def new_model_hybrid():
    input1 = Input(shape = (x.shape[1],x.shape[2]))
    en_cnn1_1 = Conv1D(64, 12,padding='same', activation= 'relu')(input1)
    en_pool = MaxPool1D()(en_cnn1_1)
    en_cnn1 = Conv1D(128, 12,padding='same', activation= 'relu')(en_pool)
    en_lstm1_1 = LSTM(128, return_sequences = True)(en_cnn1)
    en_lstm1 = LSTM(64, return_sequences = True)(en_lstm1_1)
    en_lstm2 = LSTM(64)(en_lstm1)
    repeat = RepeatVector(x.shape[1])(en_lstm2)
    conc = layers.Add()([repeat,en_cnn1_1])
    #conc = layers.Add()([en_lstm1,de_lstm1])
    de_lstm1 = LSTM(128, return_sequences = True)(conc)


    out = LSTM(x.shape[2],return_sequences = True)(de_lstm1)
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    model = Model(input1, out)
    model.compile(optimizer = 'adam', loss = 'mse', metrics = metrics)
    return model






def to_dt2(prediction_data, test_data,indx, save = False):
    noise = 91.2
    pressure = 53.98
    columns = ['Speed','Noise','Pressure']
    index = np.asarray(list(range(201))) * 0.1

    predictions = dc(prediction_data)
    test = dc(test_data)

    predictions[:,1] = predictions[:,1]*91.2
    test[:,1] = test[:,1]*91.2
    predictions[:,2] = predictions[:,2]*53.98
    test[:,2] = test[:,2]*53.98

    predictions[predictions<0.001] = 0
    df_pred = pd.DataFrame(predictions,columns=columns, index = index)
    df_real = pd.DataFrame(test,columns=columns, index = index)
    fig, axes = plt.subplots(figsize=(9,13), nrows=3, ncols=1, sharex=True, tight_layout= True)

    dist_sp = df_real['Speed'].loc[np.abs(df_real['Speed'] - df_pred['Speed']) > df_pred['Speed'].max() * 0.15]
    dist_noise = df_real['Noise'].loc[df_real['Noise'] > 69]
    dist_pressure = df_real['Pressure'].loc[np.abs(df_real['Pressure'] - df_pred['Pressure']) > 1.9]
    li = [dist_sp,dist_noise,dist_pressure ]
    for i in range(len(df_pred.columns)):
        axes[i].plot(df_real[df_real.columns[i]], label ='Actual_Data', color = 'b', linewidth = 3)
        axes[i].plot(df_pred[df_pred.columns[i]], label ='Reconstruction', color = 'g')
        axes[i].set_title(df_pred.columns[i])
        axes[i].fill_between(df_pred.index, df_real[df_real.columns[i]],df_pred[df_pred.columns[i]], color = 'gray', label = 'error')
        axes[i].plot(li[i], 'x', label = 'Above_Thresholds', color = 'r')
        axes[i].legend(loc = 'upper right')
    if save:
        plt.savefig('{}error.png'.format(indx))
    plt.show()


    '''f = 0
    f1 = 0
    f2 = 0
    f3 = 0
    f4 = 24
    f5 = 25
    f6 = 8
    f7 = 25'''

def metr(prediction,tst):
    mse_speed = metrics.mean_squared_error(prediction[:,:,0], tst[:,:,0])
    mae_speed = metrics.mean_absolute_error(prediction[:,:,0], tst[:,:,0])

    mse_noise_norm = metrics.mean_squared_error(prediction[:,:,1], tst[:,:,1])
    mae_noise_norm = metrics.mean_absolute_error(prediction[:,:,1], tst[:,:,1])
    mse_noise = metrics.mean_squared_error(prediction[:,:,1]*91.2, tst[:,:,1]*91.2)
    mae_noise = metrics.mean_absolute_error(prediction[:,:,1]*91.2, tst[:,:,1]*91.2)  


    mse_pressure_norm = metrics.mean_squared_error(prediction[:,:,2], tst[:,:,2])
    mae_pressure_norm = metrics.mean_absolute_error(prediction[:,:,2], tst[:,:,2])
    mse_pressure = metrics.mean_squared_error(prediction[:,:,2]*53.98, tst[:,:,2]*53.98)
    mae_pressure = metrics.mean_absolute_error(prediction[:,:,2]*53.98, tst[:,:,2]*53.98)
    
    print('MSE Speed = {}, MAE Speed = {}, \n \
         MSE noise normalized = {}, MSE noise = {},\n MAE noise normalized = {}, MAE noise = {}, \n \
            MSE pressure normalized = {}, MSE pressure = {}, \n MAE pressure normalized = {}, MAE pressure = {}, '.format(mse_speed,mae_speed,
             mse_noise_norm,mse_noise,mae_noise_norm,mae_noise, 
             mse_pressure_norm,mse_pressure,mae_pressure_norm,mae_pressure))

def error(recon, actual):
    mean_errors = []
    errors = tf.keras.losses.mean_absolute_error(recon,actual)
    for i in range(len(recon)):
        mean_errors.append(np.mean(errors[i]))
    return mean_errors

def error_mse(recon, actual):
    mean_errors = []
    errors = tf.keras.losses.mean_squared_error(recon,actual)
    for i in range(len(recon)):
        mean_errors.append(np.mean(errors[i]))
    return mean_errors

def plot_errors(error_norm, error_dummy, to_save = False, title = ''):
    plt.grid()
    plt.hist(error_norm, bins = 200, alpha=0.7)
    plt.hist(error_dummy, bins = 200, alpha=0.7)
    plt.legend(['Normal','Anomalies'])
    plt.xlabel('Mean reconstruction error')
    plt.ylabel('Samples')
    plt.title(title)
    plt.annotate('mean = {}\nstd = {}'.format(np.round(np.mean(error_norm), 6), np.round(np.std(error_norm), 6)), xy=(.6,.65),xycoords = 'axes fraction', c = 'b')
    plt.annotate('mean = {}\nstd = {}'.format(np.round(np.mean(error_dummy), 6), np.round(np.std(error_dummy), 6)), xy=(.6,.52),xycoords = 'axes fraction', c = 'tab:orange')
    if to_save:
        plt.savefig('{}_anomalies'.format(title), dpi = 220)
    plt.show()
    print('Mean error normal data = {} \n \
        Mean error dummy data = {}'.format(np.mean(error_norm), np.mean(error_dummy)))


def find_threshold(loss1,loss2,labels):
    mean1 = np.mean(loss1)
    mean2 = np.mean(loss2)
    losses = np.concatenate([loss1,loss2])
    losses_sort = dc(np.sort(losses))
    losses_thresholds = np.unique(losses_sort[(losses_sort>mean1) & (losses_sort<mean2)])
    max_acc = 0
    count = 0
    max_count = int(0.1*len(losses_thresholds))
    for i in range(len(losses_thresholds)):
        predictions = tf.math.less(losses, losses_thresholds[i])
        acc = np.round(accuracy_score(labels, predictions), 5)
        if acc > max_acc:
            max_acc = acc
            acc_threshold = losses_thresholds[i]
            count = 0
            best_predictions = predictions
        else:
            count +=1
            if count >max_count:
                break
    
    predictions = tf.math.less(losses, acc_threshold)
    print("Accuracy = {}".format(accuracy_score(labels, predictions)))
    print("Precision = {}".format(precision_score(labels, predictions)))
    print("Recall = {}".format(recall_score(labels, predictions)))
    return acc_threshold, best_predictions 
        


def to_dt3(prediction_data,prediction_anomalies, indx, save = False, ae = ''):
    noise = 91.2
    pressure = 53.98
    columns = ['Speed','Noise','Pressure']
    columns2 = ['Speed_anomalies','Noise_anomalies','Pressure_anomalies']
    index = np.asarray(list(range(201))) * 0.1


    predictions = dc(prediction_data[indx])
    tst = dc(test[indx])
    anomalies = dc(prediction_anomalies[indx])
    dummy = dc(dummys_array[indx])

    predictions[:,1] = predictions[:,1]*91.2
    tst[:,1] = tst[:,1]*91.2
    predictions[:,2] = predictions[:,2]*53.98
    tst[:,2] = tst[:,2]*53.98

    anomalies[:,1] = anomalies[:,1]*91.2
    dummy[:,1] = dummy[:,1]*91.2
    anomalies[:,2] = anomalies[:,2]*53.98
    dummy[:,2] = dummy[:,2]*53.98

    predictions[predictions<0] = 0
    anomalies[anomalies<0] = 0

    error_speed_normal = metrics.mean_squared_error(predictions[:,0], tst[:,0])
    error_noise_normal = metrics.mean_squared_error(predictions[:,1], tst[:,1])
    error_pressure_normal = metrics.mean_squared_error(predictions[:,2], tst[:,2])

    error_speed_anomal = metrics.mean_squared_error(anomalies[:,0], dummy[:,0])
    error_noise_anormal = metrics.mean_squared_error(anomalies[:,1], dummy[:,1])
    error_pressure_anormal = metrics.mean_squared_error(anomalies[:,2], dummy[:,2])
    
    increase_speed = int(((error_speed_anomal-error_speed_normal)/error_speed_normal)*100)
    increase_noise = int(((error_noise_anormal-error_noise_normal)/error_noise_normal)*100)
    increase_pressure = int(((error_pressure_anormal-error_pressure_normal)/error_pressure_normal)*100)

    df_pred = pd.DataFrame(predictions,columns=columns, index = index)
    df_real = pd.DataFrame(tst,columns=columns, index = index)
    df_pred_a = pd.DataFrame(anomalies,columns=columns2, index = index)
    df_real_a = pd.DataFrame(dummy,columns=columns2, index = index)

    df_pred = pd.concat([df_pred,df_pred_a], axis = 1)
    df_real = pd.concat([df_real,df_real_a], axis = 1)


    dist_sp = df_real['Speed'].loc[np.abs(df_real['Speed'] - df_pred['Speed']) > df_pred['Speed'].max() * 0.15]
    dist_noise = df_real['Noise'].loc[df_real['Noise'] > 69]
    dist_pressure = df_real['Pressure'].loc[np.abs(df_real['Pressure'] - df_pred['Pressure']) > 1.9]

    dist_sp2 = df_real_a['Speed_anomalies'].loc[np.abs(df_real_a['Speed_anomalies'] - df_pred_a['Speed_anomalies']) > df_pred_a['Speed_anomalies'].max() * 0.15]
    dist_noise2 = df_real_a['Noise_anomalies'].loc[df_real_a['Noise_anomalies'] > 69]
    dist_pressure2 = df_real_a['Pressure_anomalies'].loc[np.abs(df_real_a['Pressure_anomalies'] - df_pred_a['Pressure_anomalies']) > 1.9]
    
    li = [dist_sp,dist_noise,dist_pressure, dist_sp2,dist_noise2,dist_pressure2]

    errors = [error_speed_normal, error_noise_normal, error_pressure_normal,error_speed_anomal, error_noise_anormal, error_pressure_anormal]

    increases = ['','','',increase_speed,increase_noise, increase_pressure]

    fig, axes = plt.subplots(figsize=(15,6), nrows=2, ncols=3, sharex=True, tight_layout= True)
    plot_columns = ['Speed', 'Noise', 'Pressure', 'Speed with\nanomalies', 'Noise with\nanomalies', 'Pressure with\nanomalies']
    axes = axes.ravel(order='C')
    y_labels = ['m/s','dB','bar','m/s','dB','bar']
    for i in range(6):
        axes[i].plot(df_real[df_real.columns[i]], label ='Actual', color = 'b', linewidth = 3)
        axes[i].plot(df_pred[df_pred.columns[i]], label ='Reconstruction', color = 'g')
        if i <3:
            axes[i].annotate('MAE = {}'.format(np.round(errors[i], 7)), xy=(.68,.22),xycoords = 'axes fraction')
        else:
            axes[i].annotate('MAE = {}\nError increased {}%'.format(np.round(errors[i], 7), increases[i]), xy=(.66,.22),xycoords = 'axes fraction')
        axes[i].set_title(plot_columns[i])
        axes[i].fill_between(df_pred.index, df_real[df_real.columns[i]],df_pred[df_pred.columns[i]], color = 'gray', label = 'Error')
        axes[i].plot(li[i], 'x', label = 'Above_Thresholds', color = 'r')
        axes[i].legend(loc = 'upper right')
        axes[i].set_ylabel(y_labels[i])
        if i > 2:
            axes[i].set_xlabel('TIME')
    plt.suptitle(ae, fontsize = 19)
    if save:
        plt.savefig('{}{}error.png'.format(indx,ae))
    plt.show()


# def new_model():
#     input1 = layers.Input(shape = (x.shape[1],x.shape[2]))
#     en_lstm1_1 = LSTM(128, return_sequences = True)(input1)
#     en_lstm1_2 = LSTM(96, return_sequences = True)(en_lstm1_1)
#     en_lstm1 = LSTM(64, return_sequences = True)(en_lstm1_2)
#     en_lstm2 = LSTM(64)(en_lstm1)
#     repeat = RepeatVector(x.shape[1])(en_lstm2)
#     conc = layers.Add()([repeat,en_lstm1])
#     de_lstm1 = LSTM(96, return_sequences = True)(conc)
#     conc = layers.Add()([de_lstm1,en_lstm1_2])
#     de_lstm1 = LSTM(128, return_sequences = True)(conc)


#     out = LSTM(x.shape[2],return_sequences = True)(de_lstm1)
#     metrics = [tf.keras.metrics.MeanAbsoluteError()]
#     model = Model(input1, out)
#     model.compile(optimizer = 'adam', loss = 'mse', metrics = metrics)
#     return model


def new_model_dense():
    units = 32*201
    input1 = Input(shape = (x.shape[1],x.shape[2]))
    enc1 = Dense(128, activation = 'relu')(input1)
    enc2 = Dense(64, activation = 'relu')(enc1)
    enc2 = Dense(32, activation = 'relu')(enc2)
    flat = layers.Flatten()(enc2)
    enc2 = Dense(1024, activation = 'relu')(flat)
    z = Dense(128, activation = 'relu')(enc2)
    dec1 = Dense(1024, activation = 'relu')(z)
    dec2 = Dense(units, activation = 'relu')(dec1)
    res = layers.Reshape((201,32))(dec2)
    dec2 = Dense(64, activation = 'relu')(res)
    dec2 = Dense(128, activation = 'relu')(dec2)
    out = Dense(3, activation = 'relu')(dec2)


    metrics = [tf.keras.metrics.MeanAbsoluteError()]
    model = Model(input1, out)
    model.compile(optimizer = 'adam', loss = 'mse', metrics = metrics)
    return model