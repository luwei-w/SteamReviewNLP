import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

num_xneg_score_1 = len(xneg.loc[xneg["weighted_vote_score"]<1/3])
num_xneg_score_2 = len(xneg.loc[(1/3 <= xneg["weighted_vote_score"]) & (xneg["weighted_vote_score"]<2/3)])
num_xneg_score_3 = len(xneg.loc[(2/3<=xneg["weighted_vote_score"]) & (xneg["weighted_vote_score"]<=1)])

num_neg_score_1 = len(neg.loc[neg["weighted_vote_score"]<1/3])
num_neg_score_2 = len(neg.loc[(1/3 <= neg["weighted_vote_score"]) & (neg["weighted_vote_score"]<2/3)])
num_neg_score_3 = len(neg.loc[(2/3<=neg["weighted_vote_score"]) & (neg["weighted_vote_score"]<=1)])

num_neural_score_1 = len(neural.loc[neural["weighted_vote_score"]<1/3])
num_neural_score_2 = len(neural.loc[(1/3 <= neural["weighted_vote_score"]) & (neural["weighted_vote_score"]<2/3)])
num_neural_score_3 = len(neural.loc[(2/3<=neural["weighted_vote_score"]) & (neural["weighted_vote_score"]<=1)])

num_pos_score_1 = len(pos.loc[pos["weighted_vote_score"]<1/3])
num_pos_score_2 = len(pos.loc[(1/3 <= pos["weighted_vote_score"]) & (pos["weighted_vote_score"]<2/3)])
num_pos_score_3 = len(pos.loc[(2/3<=pos["weighted_vote_score"]) & (pos["weighted_vote_score"]<=1)])

num_xpos_score_1 = len(xpos.loc[xpos["weighted_vote_score"]<1/3])
num_xpos_score_2 = len(xpos.loc[(1/3 <= xpos["weighted_vote_score"]) & (xpos["weighted_vote_score"]<2/3)])
num_xpos_score_3 = len(xpos.loc[(2/3<=xpos["weighted_vote_score"]) & (xpos["weighted_vote_score"]<=1)])

sentiment_value = [0,1,2,3,4]
frac_recommend = [frac_recommend_xneg, frac_recommend_neg, frac_recommend_neural, frac_recommend_pos, frac_recommend_xpos]





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
fig = plt.figure()

