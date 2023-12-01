# Importing necessary libraries
import os
import json
import time
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from core.data_processor import DataLoader
from core.model import Model
import turtle
from tkinter import Tk, Canvas

# Function to plot the results
def plot_results(predicted_data, true_data, label1='True Data', label2='Prediction'):
    fig, ax = plt.subplots(facecolor='white')
    ax.plot(true_data, label=label1)
    ax.plot(predicted_data, label=label2)
    ax.legend()
    plt.show()

# Function to plot multiple results
def plot_results_multiple(predicted_data_list, true_data, prediction_len):
    fig, ax = plt.subplots(facecolor='white')
    ax.plot(true_data, label='True Data')
    for i, predicted_data in enumerate(predicted_data_list):
        padding = [None for _ in range(i * prediction_len)]
        ax.plot(padding + predicted_data, label=f'Prediction {i+1}')
    ax.legend()
    plt.show()

# Function to display a simple turtle GUI
def turtle_gui():
    root = Tk()
    root.title("Turtle GUI")
    
    canvas = Canvas(root, width=400, height=400)
    canvas.pack()

    t = turtle.RawTurtle(canvas)

    # Your turtle graphics code goes here

    root.mainloop()

# Main function
def main():
    # Load configurations from a JSON file
    configs = json.load(open('config.json', 'r'))
    
    # Ensure the model's save directory exists
    save_dir = configs['model']['save_dir']
    os.makedirs(save_dir, exist_ok=True)

    # Load data using DataLoader
    data = DataLoader(
        os.path.join('data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns']
    )

    # Initialize a neural network model
    model = Model()
    model.build_model(configs)

    # Prepare training data
    x_train, y_train = data.get_train_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # Train the model using a generator
    model.train_generator(
        data_gen=data.generate_train_batch(
            seq_len=configs['data']['sequence_length'],
            batch_size=configs['training']['batch_size'],
            normalise=configs['data']['normalise']
        ),
        epochs=configs['training']['epochs'],
        batch_size=configs['training']['batch_size'],
        steps_per_epoch=math.ceil((data.len_train - configs['data']['sequence_length']) / configs['training']['batch_size']),
        save_dir=save_dir
    )

    # Prepare testing data
    x_test, y_test = data.get_test_data(
        seq_len=configs['data']['sequence_length'],
        normalise=configs['data']['normalise']
    )

    # Make predictions using the trained model
    predictions = model.predict_sequences_multiple(x_test, configs['data']['sequence_length'], configs['data']['sequence_length'])

    # Plot the results
    plot_results_multiple(predictions, y_test, configs['data']['sequence_length'])

    # Example usage of Turtle graphics and GUI
    turtle_gui()

if __name__ == '__main__':
    main()
