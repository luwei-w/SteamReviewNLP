import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


LOW = 1/3
HIGH = 2/3


pre = pd.read_csv("../dataset/svm.csv")
pred = pre["prediction"].to_numpy()

info = pd.read_csv("../dataset/cleaned_data.csv")
value = info["weighted_vote_score"].to_numpy()

if_reco=info["voted_up"].to_numpy()#1-recommend, 0-not recommend

info["sentiment"] = pred

xneg = info.loc[info["sentiment"] == 0]
neg = info.loc[info["sentiment"]==1]
neural = info.loc[info["sentiment"]==2]
pos = info.loc[info["sentiment"]==3]
xpos = info.loc[info["sentiment"]==4]


len_xneg = len(xneg)
len_neg = len(neg)
len_neural = len(neural)
len_pos = len(pos)
len_xpos = len(xpos)

num_recommend_xneg = len(xneg.loc[xneg["voted_up"] == 1])
num_recommend_neg = len(neg.loc[neg["voted_up"] == 1])
num_recommend_neural =len(neural.loc[neural["voted_up"] == 1])
num_recommend_pos = len(pos.loc[pos["voted_up"]==1])
num_recommend_xpos = len(xpos.loc[xpos["voted_up"]==1])

frac_recommend_xneg = round(num_recommend_xneg/len_xneg, 2)
frac_recommend_neg = round(num_recommend_neg/len_neg, 2)
frac_recommend_neural = round(num_recommend_neural/len_neural, 2)
frac_recommend_pos = round(num_recommend_pos/len_pos, 2)
frac_recommend_xpos = round(num_recommend_xpos/len_xpos, 2)

num_xneg_score_1 = len(xneg.loc[xneg["weighted_vote_score"]<LOW])
num_xneg_score_2 = len(xneg.loc[(LOW <= xneg["weighted_vote_score"]) & (xneg["weighted_vote_score"]<HIGH)])
num_xneg_score_3 = len(xneg.loc[(HIGH<=xneg["weighted_vote_score"]) & (xneg["weighted_vote_score"]<=1)])

num_neg_score_1 = len(neg.loc[neg["weighted_vote_score"]<LOW])
num_neg_score_2 = len(neg.loc[(LOW <= neg["weighted_vote_score"]) & (neg["weighted_vote_score"]<HIGH)])
num_neg_score_3 = len(neg.loc[(HIGH<=neg["weighted_vote_score"]) & (neg["weighted_vote_score"]<=1)])

num_neural_score_1 = len(neural.loc[neural["weighted_vote_score"]<LOW])
num_neural_score_2 = len(neural.loc[(LOW <= neural["weighted_vote_score"]) & (neural["weighted_vote_score"]<HIGH)])
num_neural_score_3 = len(neural.loc[(HIGH<=neural["weighted_vote_score"]) & (neural["weighted_vote_score"]<=1)])

num_pos_score_1 = len(pos.loc[pos["weighted_vote_score"]<LOW])
num_pos_score_2 = len(pos.loc[(LOW <= pos["weighted_vote_score"]) & (pos["weighted_vote_score"]<HIGH)])
num_pos_score_3 = len(pos.loc[(HIGH<=pos["weighted_vote_score"]) & (pos["weighted_vote_score"]<=1)])

num_xpos_score_1 = len(xpos.loc[xpos["weighted_vote_score"]<LOW])
num_xpos_score_2 = len(xpos.loc[(LOW <= xpos["weighted_vote_score"]) & (xpos["weighted_vote_score"]<HIGH)])
num_xpos_score_3 = len(xpos.loc[(HIGH<=xpos["weighted_vote_score"]) & (xpos["weighted_vote_score"]<=1)])

score = np.array([[num_xneg_score_3, num_neg_score_3, num_neural_score_3, num_pos_score_3, num_xpos_score_3],
	[num_xneg_score_2, num_neg_score_2, num_neural_score_2, num_pos_score_2, num_xpos_score_2],
	[num_xneg_score_1, num_neg_score_1, num_neural_score_1, num_pos_score_1, num_xpos_score_1]])

print(xpos.loc[(HIGH<=xpos["weighted_vote_score"]) & (xpos["weighted_vote_score"]<=1)]["weighted_vote_score"])

sentiment_value = [0,1,2,3,4]
weighted_vote_score = [1, 0.5, 0]
frac_recommend = [frac_recommend_xneg, frac_recommend_neg, frac_recommend_neural, frac_recommend_pos, frac_recommend_xpos]

#bar plot of (sentiment value, percentage of recommendation)
fig = plt.figure()
ax = fig.add_subplot(111)
width = 0.5
plt.bar(sentiment_value, frac_recommend, width)

plt.xticks(np.arange(0,5,step = 1))
plt.xlabel('Sentiment Value')
plt.ylabel('Percentage of Recommendation')
for i,j in zip(sentiment_value,frac_recommend):
    ax.annotate(str(j),xy=(i-0.1,j))
plt.show()


#heatmap

fig, ax = plt.subplots()
im = ax.imshow(score)

# We want to show all ticks...
ax.set_xticks(np.arange(len(sentiment_value)))
ax.set_yticks(np.arange(len(weighted_vote_score)))
# ... and label them with the respective list entries
ax.set_xticklabels(sentiment_value)
ax.set_yticklabels(weighted_vote_score)
ax.set_xlabel("Sentiment Value")
ax.set_ylabel("Weighted Vote Score")

# Loop over data dimensions and create text annotations.
for i in range(len(weighted_vote_score)):
	for j in range(len(sentiment_value)):
		text = ax.text(j,i, score[i, j],
                       ha="center", va="center", color="w")

ax.set_title("The number of reviews")
fig.tight_layout()
plt.show()
