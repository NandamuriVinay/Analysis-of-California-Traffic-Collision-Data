import csv
import json
import sys
from math import e
from pprint import pprint

# import numpy as np
import numpy as np
from pyspark import SparkContext
import time

import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
#import termplotlib as plt
if __name__=="__main__":
    start = time.time()
    print("\n\nTesting your code\n")

    filename_poll= sys.argv[1]

    sc = SparkContext("local", "main app")
    sc.setLogLevel('WARN')

    rdd_poll = sc.textFile(filename_poll,32)
    rdd_poll = rdd_poll.mapPartitions(lambda x: csv.reader(x))
    broadcastHeader = sc.broadcast(rdd_poll.first())

    ### data filtering for pollution data

    # filter the first row (attribute names) and take the imp coloumns
    rdd_poll=rdd_poll.filter(lambda x: x != broadcastHeader.value).map(lambda x: ((x[5],x[6],x[8]), (x[13],x[18],x[25],x[28]))).distinct()

    #filter out the null valued coloumns
    rdd_poll=rdd_poll.filter(lambda x : x[1][0]!=''  and x[1][1]!='' and x[1][2]!='' and x[1][3]!='' and x[0][0]=='California')

    # convert the string values to float
    rdd_poll=rdd_poll.map(lambda x : (x[0],(float(x[1][0]),float(x[1][1]),float(x[1][2]),float(x[1][3]))))

    #print(rdd_poll.take(4))




    ###########################  prediction of AQI (Air Quality Index)for air pollutants #############################
    def sorted_data(data):
        region=data[0]
        sort_data=sorted(data[1],key=lambda x: x[0])
        x=[]
        for s in sort_data:
            x.append(s[1])
        return region,x



    rdd_poll=rdd_poll.map(lambda x : ((x[0][0],x[0][1]),(x[0][2],x[1][0]))).groupByKey().mapValues(list).map(sorted_data)
    #print(rdd_poll.take(1))
    rdd_poll= rdd_poll.filter(lambda x: x[0][1] == 'Los Angeles')
    #print('************* Los Angeles *****************')
    print(rdd_poll.take(1)[0])
    
    one_county=rdd_poll.take(1)[0]




    def forecast(data):
        county = data[0]
        dataset = data[1]
        print(len(dataset))
        epochs= 20

        # reframed = series_to_supervised(dataset, 8, 8)
        # values = reframed.values
        X = []
        y = []
        for i in range(len(dataset)-15):
            X.append([dataset[j] for j in range(i, i+15)])
            # y.append([dataset[j] for j in range(i+10,i+15)])
        train = np.array(X[:-3000])
        test = np.array(X[-3000:])
        #   n_obs = 8 * 13
        train_X, train_y = train[:, :-5], train[:, -5:]
        test_X, test_y = test[:, :-5], test[:, -5:]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
        model = Sequential()
        model.add(LSTM(10, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(5))
        model.compile(loss='mse', optimizer='adam')
        # fit network
        history = model.fit(train_X, train_y, epochs=epochs, batch_size=4, validation_data=(test_X, test_y), verbose=2, shuffle=False)
        # plot history
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
        yhat = model.predict(test_X)
        avg_preds = {} 
        for idx, pred8 in enumerate(yhat):
            for ind, pred in enumerate(pred8):
                val = avg_preds.get(ind + idx, 0)
                avg_preds[ind+idx] = pred
        pred = [v for k, v in avg_preds.items()]
        return county, pred

    pred = forecast(one_county)
    #print('********************* predictions **************************')
    print(pred)
    plt.subplots(figsize=(20,8))
    plt.plot(one_county[1][-3000:])
    plt.plot(pred[1])
    plt.show(block=True)

    print("############   total time in secs :", time.time() - start)


