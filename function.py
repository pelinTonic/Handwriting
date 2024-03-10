import tkinter as tk
from tkinter import END
from tkinter.filedialog import askopenfilename
import os
import cv2
from tkinter import messagebox
import numpy as np
import keras 
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt 
from PIL import Image, ImageTk
from tkinter import filedialog
import shutil

def find_numbers(imgs):

    img = cv2.imread(imgs, cv2.IMREAD_GRAYSCALE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    value, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []

    for contour in contours:
        
         
        area = cv2.contourArea(contour)


        if area < 100:
            continue
    
        x, y, w, h = cv2.boundingRect(contour)

    
        roi = img[y:y+h, x:x+w]

        roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
        
        rois.append((x, roi))
    rois.sort(key=lambda x: x[0])

    return [roi for _, roi in rois]

def open_file(file_path): 
    if file_path == "":
        pass
    else:
        file_paths = get_image_paths(file_path)
    
        return file_paths

def move_file(source_filepath, prediction, file_list, filepaths):

    destination_path = f"Slike\{prediction}"
    
    try:
        shutil.move(source_filepath, destination_path)
        file_list.update_list(open_file(filepaths))
    except Exception as e:
        print(f"An error occurred: {e}")

def wrong(filepath, file_list, filepaths):

    wrong_window = tk.Tk()
    wrong_window.title("Correction")

    wrong_window_label_frame = tk.LabelFrame(wrong_window)
    wrong_window_label_frame.pack(padx=5, pady=5)

    label = tk.Label(wrong_window_label_frame, text = "Unesite točan  broj: ")
    label.grid(row = 0, column = 0, padx=5, pady=5)

    number_entry = tk.Entry(wrong_window_label_frame)
    number_entry.grid(row=0, column=1, padx=5, pady=5)

    correct_button = tk.Button(wrong_window, text="Make correction", command= lambda: (move_file(filepath, number_entry.get(), file_list, filepaths),wrong_window.destroy()))
    correct_button.pack(padx=5, pady=5)

    wrong_window.mainloop()

def previous_picture(directory, photo_label_text, photo_label):
   
    photo_paths = get_image_paths(directory)
    if photo_label_text in photo_paths:

        try: 
            index = photo_paths.index(photo_label_text)
            image = Image.open(photo_paths[index-1])


        except IndexError:
            image = Image.open(photo_paths[0])

        image = Image.open(photo_paths[0])

    photo = ImageTk.PhotoImage(image)
    photo_label.config(image=photo, text = f"{photo_paths[0]}")
    photo_label.image = photo
    
def next_picture(directory, photo_label_text, photo_label):
  
    photo_paths = get_image_paths(directory)
    if photo_label_text in photo_paths:

        try: 
            index = photo_paths.index(photo_label_text)
            image = Image.open(photo_paths[index+1])

        except IndexError:
            image = Image.open(photo_paths[0])
    else:
        image = Image.open(photo_paths[0])
    photo = ImageTk.PhotoImage(image)
    photo_label.config(image=photo, text = f"{photo_paths[0]}")
    photo_label.image = photo

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
    confidence = min(predicted_percentages)
    predicted_number = predict_number(photo_entry)

    numbers = []
    for number in predicted_number:
        number = number.tolist()
        numbers.extend(number)

    strings = [str(num) for num in numbers]
    numbers_string = "".join(strings)

    label.config(text=f"Predicted number: {numbers_string} with {confidence}% confidence")

def predict_number(photo_path: str) -> np.ndarray:
    """
    This function predicts the digit (0-9) in a given image.

    Parameters:
    photo_path (str): The file path to the image.

    Returns:
    predicted_labels (List[np.ndarray]): The predicted labels for the digits in the image.

    The function works as follows:
    - Reads the image from the provided path in grayscale.
    - Resizes the image.
    - Loads a pre-trained keras model.
    - Makes a prediction using the model.
    - Returns the label of the digit with the highest confidence.
    """

    numbers = find_numbers(photo_path)
    predicted_labels = []
    for number in numbers:
        cv2.imshow('Result',number)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        number = resize_img(number)
        number = np.expand_dims(number, axis=0)
        model = keras.models.load_model("0.9380378723144531.keras")
        labels = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(labels)
        predictions = model.predict(number)
        predicted_label = label_encoder.inverse_transform(np.argmax(predictions, axis=1))
        predicted_labels.append(predicted_label)

    return predicted_labels

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
    
    numbers = find_numbers(photo_path)
    confidence_labels = []

    for number in numbers:
        resized_img = resize_img(number)
        resized_img = np.expand_dims(resized_img, axis=0)
        model = keras.models.load_model("0.9380378723144531.keras")
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        label_encoder = LabelEncoder()
        label_encoder.fit_transform(labels)
        predictions = model.predict(resized_img)
        num_classes = predictions.shape[1]
        confidence_percentages = [predictions[0][i] * 100 for i in range(num_classes)]
        result_dict = {label: confidence for label, confidence in zip(labels, confidence_percentages)}

        confidence_labels.append(max(result_dict.values()))

    return confidence_labels

def resize_img(photo: np.ndarray):
    """
    Resizes an image and normalizes its pixel values.

    This function takes a numpy array representation of an image as input, resizes it to 28x28 pixels using OpenCV's resize function, normalizes the pixel values by dividing by 255.0, and adds an extra dimension to the array.

    Args:
        photo (np.ndarray): The original image represented as a numpy array.

    Returns:
        np.ndarray: The resized and normalized image with an extra dimension added.
    """

    resized_img = cv2.resize(photo, (32, 32))
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
            if photo_paths == []:
                error("U mapi ne postoje datoteke")
            else:
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
