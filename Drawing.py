import cv2
import PIL
import numpy as np
import random
import os
from matplotlib import cm
from PIL import Image, ImageDraw
from tkinter import *

width = 250
height = 250
active = False
sample_dir = 'handwriting_samples'

if not os.path.isdir(sample_dir):
    os.mkdir(sample_dir)

def clicked(event):
    global active
    active = not active

def reset():
    cv.delete("all")
    draw.rectangle((0,0,250,250), fill=(255,255,255))

def save_image():
    filename = f'{sample_dir}/sample{random.getrandbits(64)}.png'
    image.save(filename)
    numpy_image = cv2.imread(filename)
    resized_numpy = cv2.resize(numpy_image, dsize=(28,28), interpolation=cv2.INTER_CUBIC)
    resized_image = PIL.Image.fromarray(resized_numpy)
    resized_image.save(filename)
    reset()

def paint(event):
    if (active):
        x1,y1 = (event.x-7), (event.y-7)
        x2,y2 = (event.x+7), (event.y+7)
        cv.create_oval(x1,y1,x2,y2,fill="black")
        draw.ellipse([x1,y1,x2,y2],fill="black")

root = Tk()

cv = Canvas(root, width=width, height=height, bg="white")
cv.pack()

image = PIL.Image.new("RGB", (width, height), (255,255,255))
draw = PIL.ImageDraw.Draw(image)

cv.pack(expand=NO, fill=BOTH)
cv.bind("<Motion>", paint)
cv.bind("<Button-1>", clicked)

button = Button(text="save", command=save_image)
button.pack()
root.mainloop()
