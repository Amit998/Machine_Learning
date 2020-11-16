import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


plt.style.use('fivethirtyeight')

frame_len= 10000

fig=plt.figure(figsize=(9,6))


def animate(i):
    data=pd.read_csv('seniment.csv')
    x=data['Time']
    y1=data["Trump"]
    y2=data["Biden"]


    if (len(y1)<= frame_len):
        plt.cla()
        plt.plot(y1,label='Donald Trump')
        plt.plot(y2,label='Joss Biden')
    else:
        plt.cla()

        plt.plot(y1[-frame_len:],label='Donald Trump')
        plt.plot(y2[-frame_len:],label='Joss Biden')
        # plt.gcf().autofmt_xdate()
    plt.legend(loc='uppser left')


    plt.tight_layout()
    # plt.show()


ani=FuncAnimation(plt.gcf(),animate,interval=100)


