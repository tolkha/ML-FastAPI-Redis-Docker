"""
Model server script that polls Redis for images to classify

Adapted from https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
"""
import base64
import json
import os
import sys
import time
from datetime import datetime, timedelta
import logging
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import redis

# Connect to Redis server
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

# Load the pre-trained Keras model 
# upload the models and weigths to the server
# load json and create model
LOG_FILENAME = 'model.json'
LOG_PATH = '/app/'
LOG_FILENAME1 = 'Weights1.h5'

filename=f'{LOG_PATH}{LOG_FILENAME}'
json_file = open(filename, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights(f"{LOG_PATH}{LOG_FILENAME1}")
print("Loaded model from disk")

model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])

def divide_data(a):
    #After preprocessing of the datas we should seperate it as input and output 

    X = a.iloc[:, 0:20].values
    y = a.iloc[:, 21:25]
    X = np.asarray(X).astype('float32')
    y = np.asarray(y).astype('float32')

    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(X)
    sc1 = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled1 = sc1.fit_transform(y)
    print("Before sending the data frame to prediction")
    # Return the Input datas
    return training_set_scaled


def database(t):
    #After preprocessing of the datas we should seperate it as input and output 
    output = pd.DataFrame()
    input_data = t
    print(type(input_data))
    output['time'] = pd.date_range(end='2022-07-02T23:59:00',freq='1min', periods= len(t))

    # To ingflux data
   
    output.set_index('time', inplace=True)
    output.index = pd.to_datetime(output.index, unit='s')


 # print(output['time'])


    influx = {}
    for column in set(input_data.columns):
        #print('Creating table: '+ column)
        influx[column] = pd.DataFrame()
        influx[column]['time'] = output.index
        influx[column]['installation'] = ['testing']*len(output)
        influx[column]['value'] = input_data[column]
        (influx[column]).set_index('time', inplace=True)
        (influx[column]).index = pd.to_datetime((influx[column]).index, unit='s')
        print(column)
        print(influx[column])

    from influxdb_client import InfluxDBClient
    step = 1000
    with InfluxDBClient(url='http://192.168.1.154:8086', token='', org='-') as client: # http://127.0.0.1:8086 http://localhost:8086  
        with client.write_api() as influx_conn:
            for parameter in influx.keys():
                print( f'Working with:{parameter}')
                for i in range(1, len(influx[parameter]), step):
                    print(f'  Slice {i} to {i+step} of {len(influx[parameter])}')
                    influx_conn.write('_internal/monitor', record=(influx[parameter].iloc[i:i+step]), data_frame_measurement_name=parameter, data_frame_tag_columns=['installation'])
                    #sleep(0.2) Test/autogen can be used if you want only observe the values you write in here
    return input


def classify_process():
    # Continually poll for new data to prediction
    while True:
        # Pop off multiple data from Redis queue atomically
        with db.pipeline() as pipe:
            pipe.lrange(os.environ.get("DATA_QUEUE"), 0, int(os.environ.get("BATCH_SIZE")) - 1)
            pipe.ltrim(os.environ.get("DATA_QUEUE"), int(os.environ.get("BATCH_SIZE")), -1)
            queue, _ = pipe.execute()

        dataIDs = []
        batch = None
        for q in queue:
            print(type(q))
            q1 = q.decode("utf-8")
            q2 =json.loads(q1)
            print(type(q2))
            df= pd.DataFrame(eval(q2["data"]))
            print("Before sending the divide_Data")
            datas = divide_data(df)

            # Check to see if the batch list is None
            if batch is None:
                batch = datas

            # Otherwise, stack the data
            else:
                batch = np.vstack([batch, datas])

            # Update the list of data IDs
            dataIDs.append(q2["id"])

        # Check to see if we need to process the batch
        if len(dataIDs) > 0:
            # Classify the batch
            print("* Batch size: {}".format(batch.shape))
            results = model.predict(batch)
            print("* Results size: {}".format(results.shape))   
            print(results)
            print(type(results))
            df = pd.DataFrame.from_records(results)
            print(type(df))
            df1 = database(df)
            print(df1)
            json_str = df.to_json(orient = 'columns')
            dataIDs_str = ' '.join([str(elem) for elem in dataIDs])
            print(type(dataIDs_str))
            print(type(json.dumps(json_str)))
            db.set(dataIDs_str, json.dumps(json_str))
        # Sleep for a small amount
        time.sleep(float(os.environ.get("SERVER_SLEEP")))

if __name__ == "__main__":
    classify_process()
