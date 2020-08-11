import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pre = pd.read_csv("../dataset/svm.csv")
pred = pre["prediction"].to_numpy()

info = pd.read_csv("../dataset/cleaned_data.csv")
value = info["weighted_vote_score"].to_numpy()

fig = plt.figure()
plt.scatter(pred, value)
plt.xticks(np.arange(0,5,step = 1))
plt.xlabel('Sentiment value')
plt.ylabel('Weighted_vote_score')
plt.show()
