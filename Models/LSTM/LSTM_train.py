from keras.models import Sequential
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout, LSTM, Activation

def LSTM_train(data):
    seq_array, label_array, lstm_test_df, sequence_length, sequence_cols = lstm_data_preprocessing(data[0], data[1], data[2])
    model_instance, history = lstm_train(seq_array, label_array, sequence_length)
    return model_instance, history, lstm_test_df, seq_array, label_array, sequence_length, sequence_cols