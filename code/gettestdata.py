import pandas as pd
import numpy as np

info = pd.read_csv("../dataset/cleansteam.csv", sep = ",", usecols = ["recommendation_id","review"], encoding = "ISO-8859-1")
info.to_csv("../dataset/test.csv", sep = ",", header = None, index = None, encoding = "ISO-8859-1")

