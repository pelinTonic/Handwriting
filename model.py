import numpy as np
import tensorflow
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from keras import models, layers, utils, callbacks
from functions import loading_and_preprocessing
import pandas as pd

test_acc_list = []
test_acc = 0
dataset_path = "Slike"  
df = pd.DataFrame()

for i in range(0, 10):
    
    test_acc_list = []
    data = loading_and_preprocessing(dataset_path)[0]
    labels = loading_and_preprocessing(dataset_path)[1]

    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    data = np.array(data)
    labels_encoded = np.array(labels_encoded)
        

    train_data, test_data, train_labels, test_labels = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)
            

    model = models.Sequential([

        layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 1)),  
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Dropout(0.13),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.70),  
        layers.Dense(len(label_encoder.classes_), activation='softmax'),
            ])
            
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)


    model.fit(train_data, train_labels, epochs=30, validation_split=0.20, callbacks=[early_stopping])
    test_loss, test_acc = model.evaluate(test_data, test_labels)

    test_acc_list.append(test_acc)
    model.save(f"Modeli/{test_acc}.keras")


column_name = f"column {0.7}"
df[column_name] = test_acc_list
df.to_excel(f"Test_file_3.xlsx")