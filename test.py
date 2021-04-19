import os
if os.path.isfile("demo.db"):
    os.remove("demo.db")
from sqlalchemy import create_engine


from sklearn.datasets import load_digits
import numpy as np
digits = load_digits().data

def display_preprocessing(x):
    return x.reshape(8, 8)

from superintendent.distributed import ClassLabeller

widget = ClassLabeller.from_images(
    connection_string="sqlite:///demo.db",
    options=range(10),
    canvas_size=(200, 200),
    display_preprocess=display_preprocessing
)