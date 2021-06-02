from neuralintents import  GenericAssistant
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import mplfinance as mpf
import pickle
import sys
import datetime as dt


def myFunction():
    pass


mappings={
    'greetings':myFunction
}

portfolio={'AAPL':20,'TSLA':5,'GS':10}

assistant=GenericAssistant('intents.json',intent_methods=mappings)

assistant.train_model()

assistant.save_model()
# assistant.request()