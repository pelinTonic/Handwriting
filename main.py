import tkinter as tk
from function import *
from PIL import Image, ImageTk
from tkinter import ttk

def open_file(file_path, label): 

    file_paths = get_image_paths(file_path)
    show_file_list(file_paths, label)

def show_main_screen():
    
    main_screen = tk.Tk()
    main_screen.title("HRS")

    photo_entry_label_frame = tk.LabelFrame(main_screen)
    photo_entry_label_frame.pack(expand=True, anchor=tk.CENTER, padx=5, pady=5)

    photo_label = tk.Label(photo_entry_label_frame,text="Fotografija: ")
    photo_entry = tk.Entry(photo_entry_label_frame, width=50)
    photo_label.grid(row=0, column=0)
    photo_entry.grid(row=0, column=1,)

    button_label_frame = tk.LabelFrame(main_screen)
    button_label_frame.pack(expand=True, anchor=tk.CENTER,padx=5, pady=5)
    photo_button = tk.Button(button_label_frame, text="Upload folder", command=lambda: select_directory(photo_entry))
    photo_button.grid(row=1, column=0, padx=5, pady=5)

    predicted_number_label= tk.Label(button_label_frame, text=" ")

    predict_button = tk.Button(button_label_frame, text="Predict", command=lambda: show_predicted_data(predicted_number_label,photo_label.cget("text")))
    predict_button.grid(row=1, column=2, padx=5, pady=5)
    predicted_number_label.grid(row=2, column=1, padx=5, pady=5)

    photo_label_frame = tk.LabelFrame(main_screen)
    photo_label_frame.pack(expand=True, anchor=tk.CENTER, padx=5, pady=5)

    supplementary_photo_label_frame_1 = tk.LabelFrame(photo_label_frame)
    supplementary_photo_label_frame_1.grid(row=0, column=0, padx=5, pady=5)

    supplementary_photo_label_frame_2 = tk.LabelFrame(photo_label_frame)
    supplementary_photo_label_frame_2.grid(row=0, column=1, padx=5, pady=5)
    
    photo_label = tk.Label(supplementary_photo_label_frame_1, image=None)
    photo_label.pack(padx=5, pady=5)

    next_button = tk.Button(supplementary_photo_label_frame_1, text="Next photo", command=lambda: next_picture(photo_entry.get(),photo_label.cget("text"), photo_label))
    next_button.pack(padx=5, pady=5)

    previous_button = tk.Button(supplementary_photo_label_frame_1, text=" Previous photo", command=lambda: previous_picture(photo_entry.get(),photo_label.cget("text"), photo_label))
    previous_button.pack(padx=5, pady=5)

    open_button = tk.Button(button_label_frame, text="Open", command=lambda: open_file(photo_entry.get(), photo_label))
    open_button.grid(row=1, column=1, padx=5, pady=5)

    correct_button = tk.Button(supplementary_photo_label_frame_2, text="Correct", command= lambda: move_file(photo_label.cget("text"),predicted_number_label.cget("text")))
    correct_button.pack(padx=5, pady=5)

    wrong_button = tk.Button(supplementary_photo_label_frame_2, text="Wrong", command= lambda: wrong(photo_label.cget("text")))
    wrong_button.pack(padx=5, pady=5)

    main_screen.mainloop()

# Real time update -  napraviti 
def show_file_list(filepath, label):
    

    file_list_screen = tk.Tk()
    file_list_screen.title("File list")
    max_width = max(len(file) for file in filepath)
    file_list = tk.Listbox(file_list_screen, selectmode=tk.SINGLE, width=max_width+2)
    file_list.delete(0, tk.END)
    file_list.pack()
    for file in filepath:
        file_list.insert(tk.END, file)

    file_list.bind("<<ListboxSelect>>", lambda event: open_image(event, file_list, label))
    file_list_screen.mainloop()  

if __name__ == "__main__":
    show_main_screen()