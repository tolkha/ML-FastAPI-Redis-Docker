"""
Web server script that exposes endpoints and pushes images to Redis for classification by model server. Polls
Redis for response from model server.

Adapted from https://www.pyimagesearch.com/2018/02/05/deep-learning-production-keras-redis-flask-apache/
"""
import base64
import io
import json
import os
import time
import uuid
import logging

import numpy as np
import pandas as pd
import redis
import json
import aiofiles

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.requests import Request

app = FastAPI()
db = redis.StrictRedis(host=os.environ.get("REDIS_HOST"))

CLIENT_MAX_TRIES = int(os.environ.get("CLIENT_MAX_TRIES"))
CLIENT_TIMEOUT = int(os.environ.get('DOCKER_CLIENT_TIMEOUT', 120))


def prepare_datas(datas):
  
    df1 = datas.dropna().reset_index(drop=True)
    df1.info()
    #print(df1.columns.tolist())
    df1['GenPwr    (kW)      '] = pd.to_numeric(df1['GenPwr    (kW)      '])
    df1_clean = df1.dropna().reset_index(drop=True)
    #print(df1_clean['GenPwr    (kW)      '].describe())  # Describe the distribution of P
    selected_columns = df1_clean[["Wind1VelX (m/s)     ", "Azimuth   (deg)     ","RotSpeed  (rpm)     ","GenSpeed  (rpm)     ","PtfmSurge (m)       ","PtfmSway  (m)       ","PtfmHeave (m)       ","PtfmRoll  (deg)     ","PtfmPitch (deg)     ","PtfmYaw   (deg)     ","TTDspFA   (m)       ","WaveDir","PropagationDir","WaveTp","WaveHS","HWindSpeed", "TwrBsFxt  (kN)      ","TwrBsFyt  (kN)      ","TTDspSS   (m)       ","OoPDefl1  (m)       ","IPDefl1   (m)       ","GenPwr    (kW)      ","ANCHTEN1  (N)       ", "ANCHTEN2  (N)       ","ANCHTEN3  \n(N)       \n"]]
    new_df = selected_columns.copy()
    return new_df



@app.get('/')
async def root():
    """ Function to test the system and do some kind of presentation. """
    return {"title": "Digital twin PV production v1",
            "subtitle": "developed in the context of SuperPV project in IREC",
            "authors": [
                {"name": "Tolga Yalçin",
                 "email": "<tyalcin@irec.cat>"
                 },
                 {"name": "Paschalia Stefanidou",
                 "email": "<pstefanidou@irec.cat>"
                 },
                {"name": "Pol Paradell Sola",
                 "email": "<pparadell@irec.cat>"}]}



@app.post("/predict")
def predict(request: Request, data_file: bytes=File(...)):
    data = {"success": False}

    if request.method == "POST":
        datas =  pd.read_csv(io.BytesIO(data_file)) # Not sure does it can read or not 
        #datas= prepare_datas(datas)
        datas = datas.to_json()
        print(datas)

        # Generate an ID for the classification then add the classification ID + data to the queue
        k = str(uuid.uuid4())
        d = {"id": k, "data": datas} # The data is the csv file cfile.csv
        ds= json.dumps(d) 
        db.rpush(os.environ.get("DATA_QUEUE"), ds)

        # Keep looping for CLIENT_MAX_TRIES times
        num_tries = 0
        while num_tries < CLIENT_MAX_TRIES:
            num_tries += 1

            # Attempt to grab the output predictions
            output = db.get(k)

            # Check to see if our model has classified the input image
            if output is not None:
                # Add the output predictions to our data dictionary so we can return it to the client
                data["predictions"] = json.loads(output)

                # Delete the result from the database and break from the polling loop
                #db.delete(k)
                break

            # Sleep for a small amount to give the model a chance to classify the input image
            time.sleep(float(os.environ.get("CLIENT_SLEEP")))

            # Indicate that the request was a success
            data["success"] = True
        else:
            raise HTTPException(status_code=400, detail="Request failed after {} tries".format(CLIENT_MAX_TRIES))

    # Return the data dictionary as a JSON response
    return data
