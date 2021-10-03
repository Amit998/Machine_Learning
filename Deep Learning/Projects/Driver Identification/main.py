import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data=pd.read_csv("full_data_test.csv")

columns2=["Long_Term_Fuel_Trim_Bank1","Intake_air_pressure","Accelerator_Pedal_value","Fuel_consumption","Torque_of_friction","Maximum_indicated_engine_torque","Engine_torque","Calculated_LOAD_value",
"Activation_of_Air_compressor","Engine_coolant_temperature","Transmission_oil_temperature","Wheel_velocity_front_left-hand","Wheel_velocity_front_right-hand","Wheel_velocity_rear_left-hand",
"Torque_converter_speed"]

from sklearn import svm
ano_det=svm.OneClassSVM(nu=0.1,kernel="rbf",gamma=0.1)
ano_det.fit(data[columns2])

classes=['A','B','C','D','E','F','G','H','I','J']

drivers=[]

for c in classes:
    drivers.append(data[data['class']==c])

dataa=[]

for c in range(len(drivers)):
    nt=0
    nv=0
    drivers[c]=drivers[c].reset_index(drop=True)
    # idxs=drivers[c][]