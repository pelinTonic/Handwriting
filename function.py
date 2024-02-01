import tkinter as tk
from tkinter import END
from tkinter.filedialog import askopenfilename
import os
import cv2
from tkinter import messagebox
import numpy as np
import keras 
from sklearn.preprocessing import LabelEncoder
import pandas
import matplotlib.pyplot as plt 
from PIL import Image, ImageTk
from tkinter import filedialog
from main import show_file_list
import shutil


    
def move_file(source_filepath, prediction):
    destination_path = f"Slike\{prediction}"

    try:
        shutil.move(source_filepath, destination_path)
    except Exception as e:
        print(f"An error occurred: {e}")

def wrong(filepath):

    wrong_window = tk.Tk()
    wrong_window.title("Correction")

    wrong_window_label_frame = tk.LabelFrame(wrong_window)
    wrong_window_label_frame.pack(padx=5, pady=5)

    label = tk.Label(wrong_window_label_frame, text = "Unesite točan  broj: ")
    label.grid(row = 0, column = 0, padx=5, pady=5)

    number_entry = tk.Entry(wrong_window_label_frame)
    number_entry.grid(row=0, column=1, padx=5, pady=5)

    correct_button = tk.Button(wrong_window, text="Make correction", command= lambda: move_file(filepath, number_entry.get()))
    correct_button.pack(padx=5, pady=5)

    wrong_window.mainloop()

def previous_picture(directory, photo_label_text, photo_label):
    
    photo_paths = get_image_paths(directory)
    if photo_label_text in photo_paths:

        try: 
            index = photo_paths.index(photo_label_text)
            image = Image.open(photo_paths[index-1])
            photo = ImageTk.PhotoImage(image)
            photo_label.config(image=photo, text = f"{photo_paths[index-1]}")
            photo_label.image = photo

        except IndexError:
            image = Image.open(photo_paths[0])
            photo = ImageTk.PhotoImage(image)
            photo_label.config(image=photo, text = f"{photo_paths[0]}")
            photo_label.image = photo,
    
def next_picture(directory, photo_label_text, photo_label):
    
    photo_paths = get_image_paths(directory)
    if photo_label_text in photo_paths:

        try: 
            index = photo_paths.index(photo_label_text)
            image = Image.open(photo_paths[index+1])
            photo = ImageTk.PhotoImage(image)
            photo_label.config(image=photo, text = f"{photo_paths[index+1]}")
            photo_label.image = photo

        except IndexError:
            image = Image.open(photo_paths[0])
            photo = ImageTk.PhotoImage(image)
            photo_label.config(image=photo, text = f"{photo_paths[0]}")
            photo_label.image = photo,

def get_image_paths(directory):

    image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
    photo_paths = []

    for file_name in os.listdir(directory):
        if any(file_name.lower().endswith(ext) for ext in image_extensions):
            photo_paths.append(os.path.join(directory, file_name))

    return photo_paths
    
def plot_bar_chart(result_dict: dict):

    """
    This function plots a bar chart of the confidence percentages for each digit.

    Parameters:
    result_dict (dict): A dictionary where the keys are the digits (0-9) and the values are the corresponding confidence percentages.

    The function works as follows:
    - Extracts the keys and values from the dictionary.
    - Plots a bar chart using matplotlib where the x-axis represents the labels (digits) and the y-axis represents the confidence percentages.
    - Sets the x-label, y-label, and title of the plot.
    - Displays the plot.
    """
        
    x_values = list(result_dict.keys())
    y_values = list(result_dict.values())

    plt.clf()
    plt.bar(x_values, y_values)
    plt.xlabel('Label')
    plt.ylabel('Confidence (%)')
    plt.title('Confidence Percentages for Each Label')
    plt.xticks(range(10))

    plt.show()

def show_predicted_data(label: tk.Label, photo_entry: str):

    """
    This function predicts the digit in an image, updates a label with the predicted digit, and displays a bar chart of confidence percentages for each digit.

    Parameters:
    label (tk.Label): The label to be updated with the predicted digit.
    photo_entry (str): The file path to the image.

    The function works as follows:
    - Calls the `predict_number_percentages` function to get the confidence percentages for each digit.
    - Calls the `predict_number` function to predict the digit in the image.
    - Updates the provided label with the predicted digit.
    - Calls the `plot_bar_chart` function to display a bar chart of the confidence percentages.
    """
     
    predicted_percentages = predict_number_percentages(photo_entry)
    predicted_number = predict_number(photo_entry)
    label.config(text=f"{predicted_number}")

    plot_bar_chart(predicted_percentages)

def predict_number(photo_path: str) -> np.ndarray:

    """
    This function predicts the digit (0-9) in a given image.

    Parameters:
    photo_path (str): The file path to the image.

    Returns:
    predicted_labels (np.ndarray): The predicted label for the digit in the image.

    The function works as follows:
    - Reads the image from the provided path in grayscale.
    - Resizes the image.
    - Loads a pre-trained keras model.
    - Makes a prediction using the model.
    - Returns the label of the digit with the highest confidence.
    """
    img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    resized_img = resize_img(img)
    resized_img = np.expand_dims(resized_img, axis=0)
    model = keras.models.load_model("0.9186046719551086.keras")
    labels = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(labels)
    predictions = model.predict(resized_img)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
    
    return predicted_labels[0]

def predict_number_percentages(photo_path: str) -> np.ndarray:

    """
    This function predicts the confidence percentages for each digit (0-9) in a given image.

    Parameters:
    photo_path (str): The file path to the image.

    Returns:
    result_dict (np.ndarray): A dictionary where the keys are the digits (0-9) and the values are the corresponding confidence percentages.

    The function works as follows:
    - Reads the image from the provided path in grayscale.
    - Resizes the image.
    - Loads a pre-trained keras model.
    - Makes a prediction using the model.
    - Calculates the confidence percentages for each class.
    - Returns a dictionary mapping each label to its corresponding confidence percentage.
    """

    img = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
    resized_img = resize_img(img)
    resized_img = np.expand_dims(resized_img, axis=0)
    model = keras.models.load_model("0.9186046719551086.keras")
    labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(labels)
    predictions = model.predict(resized_img)
    num_classes = predictions.shape[1]
    confidence_percentages = [predictions[0][i] * 100 for i in range(num_classes)]
    result_dict = {label: confidence for label, confidence in zip(labels, confidence_percentages)}

    return result_dict

def resize_img(photo: np.ndarray):
    """
    Resizes an image and normalizes its pixel values.

    This function takes a numpy array representation of an image as input, resizes it to 28x28 pixels using OpenCV's resize function, normalizes the pixel values by dividing by 255.0, and adds an extra dimension to the array.

    Args:
        photo (np.ndarray): The original image represented as a numpy array.

    Returns:
        np.ndarray: The resized and normalized image with an extra dimension added.
    """

    resized_img = cv2.resize(photo, (28, 28))
    resized_img = resized_img / 255.0
    resized_img = np.expand_dims(resized_img, axis=-1)

    return resized_img

def error(error: str):
    """
    Display a pop-up message box with the provided error message
    Args:
        error (str): The error message to be displayed in the pop-up .1113,54
    """
    messagebox.showinfo("",error)

def select_directory(photo_entry):

    directory_path = filedialog.askdirectory(title="Select a Directory")
    if directory_path:
        photo_paths = get_image_paths(directory_path)
        if photo_paths:
            photo_entry.delete(0, END)
            photo_entry.insert(tk.END, directory_path)
            return photo_paths

def upload_file(photo_entry: str):

    """
    Opens a file dialog to allow the user to select an image file, and updates a text entry widget with the file's absolute path.

    This function opens a file dialog that filters for .jpg and .png files. If the user selects a file, the function gets the absolute path of the file and updates the specified text entry widget with this path.

    Args:
        photo_entry (str): The text entry widget to be updated with the file path.

    """

    file_types = [("Jpg files", "*jpg"), ("PNG files","*png")]
    filename = tk.filedialog.askopenfilename(filetypes = file_types)
    if filename:
        filepath = os.path.abspath(filename)
        photo_entry.delete(0, END)
        photo_entry.insert(tk.END, filepath)     
        show_file_list(filepath)

def open_image(event, listbox, label):

    selected_index = listbox.curselection()
    if selected_index:

        selected_item = listbox.get(selected_index)
        image = Image.open(selected_item)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo, text = f"{selected_item}")
        label.image = photo