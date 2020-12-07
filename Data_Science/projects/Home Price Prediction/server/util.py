import json
from flask import json
import pickle
import os
import numpy as np

__locations=None
__data_columns=None
__model=None

def get_estimated_price(location,sqft,bhk,bath):
    load_saved_artifacts()
    try:
        loc_index=__data_columns.index(location.lower())
    except:
        loc_index =-1
    x=np.zeros(len(__data_columns))
    x[0]=sqft
    x[1]=bath
    x[2]=bhk
    if (loc_index >= 0):
        x[loc_index]=1
    
    return round(__model.predict([x])[0],2)

    # return __model.predict([x])



def get_location_names():
    load_saved_artifacts()
    return __locations

def load_saved_artifacts():
    print("loading saved artifacts..start...")
    global __data_columns
    global __locations
    global __model

    with open("server/artifacts/columns.json",'r') as f:
        __data_columns= json.load(f)['data_columns']
        __locations=__data_columns[3:]
    
    with open("server/artifacts/bangalore_home_price_model.pickle",'rb') as f:
        try:
            __model=pickle.load(f)
        except EOFError:
            print("Error Threw in Loading pickle data")

    
        
        # __model=pickle.load(f)
    # print("loading saved artifacts...done")


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('1st Phase JP Nagar',1000,3,3),'L')
    print(get_estimated_price('rajaji nagar',1000,3,3),'L')
    print(get_estimated_price('rajaji nagar',1000,3,4),"L")
    print(get_estimated_price('yelahanka',1000,3,3),"L")
    print(get_estimated_price('singasandra',1000,3,3),"L")
    print(get_estimated_price('ramamurthy nagar',1000,3,3),"L")