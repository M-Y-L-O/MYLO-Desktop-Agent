import os
def loadModel(path:str):
    if os.path.exists(path):
        with open(path, 'r') as file:
            binary_data = file.read()
            return binary_data
    else:
        raise FileNotFoundError("Model file not found")
    
def pickModelFile():
    from tkinter import Tk
    from tkinter import filedialog

    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select ONNX Model", filetypes=[("ONNX files", "*.onnx")])
    
    if file_path:
        return file_path
    else:
        raise FileNotFoundError("No file selected")
    
def pickCSVFile():
    from tkinter import Tk
    from tkinter import filedialog

    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select CSV dataset", filetypes=[("CSV files", "*.csv")])
    
    if file_path:
        return file_path
    else:
        raise FileNotFoundError("No file selected")
    
def pickProjectFile():
    from tkinter import Tk
    from tkinter import filedialog

    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select MYLO project", filetypes=[("MYLO projects", "*.mylo")])
    
    if file_path:
        return file_path
    else:
        raise FileNotFoundError("No file selected")

def saveProjectFilePath():
    from tkinter import Tk
    from tkinter import filedialog

    root = Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(title="Save MYLO project", defaultextension=".mylo", filetypes=[("MYLO projects", "*.mylo")])
    
    if file_path:
        return file_path
    else:
        raise FileNotFoundError("No file selected")