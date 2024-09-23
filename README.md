# Glucose_Monitor

import time
import numpy as np
import pandas as pd
import RPi.GPIO as GPIO
from tkinter import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# GPIO pin setup for MCP3008 ADC
SPI_CLK = 11  # Clock pin
SPI_MISO = 9  # Master In Slave Out
SPI_MOSI = 10 # Master Out Slave In
SPI_CS = 8    # Chip Select
MQ138_CHANNEL = 0  # Channel for MQ-138 on MCP3008

# Initialize GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SPI_CLK, GPIO.OUT)
GPIO.setup(SPI_MISO, GPIO.IN)
GPIO.setup(SPI_MOSI, GPIO.OUT)
GPIO.setup(SPI_CS, GPIO.OUT)

def read_adc(adc_channel):
    """
    Reads the value from the specified ADC channel.
    
    Parameters:
        adc_channel (int): The channel number to read (0-7).
    
    Returns:
        int: The ADC value (0-1023).
    """
    if adc_channel > 7 or adc_channel < 0:
        return -1
    GPIO.output(SPI_CS, GPIO.LOW)  # Select the ADC
    adc = adc_channel | 0x18  # Start bit + single-ended input
    adc <<= 3  # Shift to the left for transmission
    for i in range(5):  # Send 5 bits
        GPIO.output(SPI_MOSI, GPIO.HIGH if adc & 0x80 else GPIO.LOW)
        adc <<= 1
        GPIO.output(SPI_CLK, GPIO.HIGH)
        GPIO.output(SPI_CLK, GPIO.LOW)
    adc_out = 0
    for i in range(12):  # Read 12 bits of data
        GPIO.output(SPI_CLK, GPIO.HIGH)
        GPIO.output(SPI_CLK, GPIO.LOW)
        adc_out <<= 1
        if GPIO.input(SPI_MISO):
            adc_out |= 0x1
    GPIO.output(SPI_CS, GPIO.HIGH)  # Deselect the ADC
    adc_out >>= 1  # Discard the first 'null' bit
    return adc_out

# Simulated dataset generation (replace with actual data collection)
def generate_simulated_data(size=500):
    """
    Generates simulated acetone and glucose level data.
    
    Parameters:
        size (int): The number of data points to generate.
    
    Returns:
        tuple: Arrays of acetone and glucose levels.
    """
    acetone_levels = np.random.rand(size) * 10  # Simulated acetone levels (0-10)
    glucose_levels = acetone_levels * 10 + np.random.randn(size) * 5  # Simulated glucose levels
    return acetone_levels, glucose_levels

# Train the machine learning model
def train_model(X, y):
    """
    Trains a Random Forest model to predict glucose levels from acetone levels.
    
    Parameters:
        X (numpy.ndarray): Input feature array (acetone levels).
        y (numpy.ndarray): Output target array (glucose levels).
    
    Returns:
        RandomForestRegressor: Trained model.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    return model

# Load or generate training data
acetone_levels, glucose_levels = generate_simulated_data()
X = acetone_levels.reshape(-1, 1)  # Reshape for model input
y = glucose_levels

# Train the model
model = train_model(X, y)

# Tkinter GUI setup
root = Tk()
root.title("Breath-Based Glucose Monitor")
root.geometry("400x300")

# Display label for glucose estimation
glucose_label = Label(root, text="Estimated Glucose Level: -- mg/dL", font=("Helvetica", 16))
glucose_label.pack(pady=20)

def update_glucose():
    """
    Updates the glucose estimation in the GUI by reading the sensor data
    and predicting glucose levels using the trained model.
    """
    # Read the current acetone value from the sensor
    acetone_value = read_adc(MQ138_CHANNEL) / 1023.0 * 10  # Scale to 0-10
    # Predict glucose level using the trained model
    glucose_estimate = model.predict([[acetone_value]])[0]
    # Update the display label with the estimated glucose level
    glucose_label.config(text=f"Estimated Glucose Level: {glucose_estimate:.2f} mg/dL")
    # Schedule the next update
    root.after(2000, update_glucose)  # Update every 2 seconds

# Start the GUI update loop
root.after(2000, update_glucose)
root.mainloop()

# Cleanup GPIO resources on exit
GPIO.cleanup()
