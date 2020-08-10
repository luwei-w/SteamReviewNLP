# SteamReviewNLP

Dataset

1. train.csv has 11855 movie reviews. The first column is the id of this review, the second column is its sentiment value, the third column is its text. In the sentiment column, the meaning of values are: 0-very negative, 1-negative, 2-neural, 3-positvie, 4-very positve. If you train your data using this model, you should change the train dataset format as the same as "train.csv" and no header.

2. test data should have the following format. It should have two columes, the first column is ID, the second column is text, no header.

===================================

Code

In your terminal, input the following commands.

1. cd /path/to/SteamReviewNLP/code

2. python3 preprocess.py ../dataset/train.csv

3. python3 stats.py ../dataset/train-processed.csv

4. python3 svm.py

5. python3 naivebayes.py

The step 2 outputs a file called train-processed.csv. The step 3 outputs 3 files which are train-processed-freqdist-bi.pkl, train-processed-freqdist.pkl, train-processed-unique.txt. Step 4 and 5 depends on which method you choose. 

In preprocess.py, there is a boolean variable called "test_file" in the function "preprocess_csv()", when preprocessing training dataset, you should assign it "TRUE", when preprocessing test dataset, assign it "FALSE". 

We do not need to apply stats.py on test dataset. Because we only use frequency distribution of training dataset dictionary.

In svm.py and naivebayes.py, there is a global boolean variable called "TRAIN", when training the model, you should assign it "TRUE", when predicting the test dataset, you should assign it "FALSE". After prediction, svm.py will output the prediction result file "svm.csv" in folder "code", naivebayes.py will output "naivebayes.csv".
