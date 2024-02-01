import tkinter as tk
from PIL import Image, ImageTk

class FileList:

    def __init__(self, label, filepath) -> None:
        
        self.filepath = filepath
        self.label = label

    def create_widget(self):

        max_width = max(len(file) for file in self.filepath)
        self.new_window = tk.Tk()
        self.new_window.title("List of test files")

        self.filelist = tk.Listbox(self.new_window, selectmode=tk.SINGLE, width=max_width + 2)
        self.filelist.pack()

        for file in self.filepath:
            self.filelist.insert(tk.END, file)

        self.filelist.bind("<<ListboxSelect>>", self.open_image)

    def open_image(self, event):

        selected_index = self.filelist.curselection()

        if selected_index:
            selected_item = self.filelist.get(selected_index)
            image = Image.open(selected_item)
            photo = ImageTk.PhotoImage(image)
            self.label.config(image = photo, text = f"{selected_item}")
            self.label.image = photo

    def update_list(self):

        self.filelist.delete(0, tk.END)
        for file in self.filepath:
            self.filelist.insert(tk.END, file)   


        
        