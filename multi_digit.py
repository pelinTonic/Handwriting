import cv2
import numpy as np
import pandas as pd
import keras 
from sklearn.preprocessing import LabelEncoder
from function import get_image_paths


def df_to_array(df):
    col_splits_array = df.values
    col_splits_resized = cv2.resize(col_splits_array,(32,32))
    col_splits_resized = col_splits_resized / 255.0
    col_splits_resized = np.expand_dims(col_splits_resized, axis=0)

    return col_splits_resized

def define_split_parametars(mean):

    pass

def convert_image_to_df(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_df = pd.DataFrame(image)

    return image_df

def split(df, division_range):

    
    col_splits_resized_list = []
    num_cols = df.shape[1]
    
    for i in range(0, num_cols, division_range):
        col_end = min(i+division_range, num_cols)
        col_splits = df.iloc[:, i:col_end]
        col_splits_resized = df_to_array(col_splits)
        col_splits_resized_list.append(col_splits_resized)

    return col_splits_resized_list

def mean(image_df):

    column_averages = image_df.mean()
    return column_averages

def predict(array):
    
    model = keras.models.load_model("0.9666666388511658.keras")
    labels = [0,1, 2, 3, 4, 5, 6, 7, 8, 9]
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(labels)
    predictions = model.predict(array)
    predicted_labels = label_encoder.inverse_transform(np.argmax(predictions, axis=1))

    return predicted_labels

def find_numbers(imgs):

    img = cv2.imread(imgs, cv2.IMREAD_GRAYSCALE)
    img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresh = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rois = []

    for contour in contours:
        
        
        x, y, w, h = cv2.boundingRect(contour)
        x -= 5
        y -= 5
        w += 10 
        h += 10  
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
        roi = img[y:y+h, x:x+w]

        roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
        cv2.imshow('Result',roi)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        rois.append(roi)

    return rois

def multi_digit_prediction(file):

    images = get_image_paths(file)
    for img in images:
        imgs = convert_image_to_df(img)
        numbers = split(imgs,40)

        for number in numbers:
            print(predict(number))

#imgs = get_image_paths("Test_slike\Test_multi_digit")
#rois = find_numbers("Test_slike/Test_multi_digit/20873.jpg")
file = "Test_slike/Test_multi_digit/20873.jpg"
img_df = convert_image_to_df(file)

img_df.to_excel("img.xlsx")
# for img in imgs:
#     rois = find_numbers(img)
#     print(img)
# for roi in rois:

#     col_splits_resized = cv2.resize(roi,(32,32))
#     col_splits_resized = col_splits_resized / 255.0
#     col_splits_resized = np.expand_dims(col_splits_resized, axis=0)
#     print(predict(col_splits_resized))