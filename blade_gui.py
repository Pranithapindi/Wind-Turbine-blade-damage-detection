from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

model = YOLO("runs/classify/train/weights/best.pt")

def predict():

    file_path = filedialog.askopenfilename()

    results = model(file_path)

    result = results[0]

    names = result.names
    probs = result.probs.data.tolist()

    index = probs.index(max(probs))

    prediction = names[index]
    confidence = probs[index]

    label.config(text=f"{prediction} ({confidence:.2f})")

window = tk.Tk()
window.title("Wind Blade Fault Detection")

btn = tk.Button(window,text="Upload Image",command=predict)
btn.pack(pady=20)

label = tk.Label(window,text="Result will appear here")
label.pack(pady=20)

window.mainloop()