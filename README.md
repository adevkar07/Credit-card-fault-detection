# Credit-card-fault-detection
I worked on this problem statement so as to be able to recognize fraudulent credit card transactions from overall transactions. The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions which was one of the challenge.

I used 2 techniques for this, one was by Anomaly Detection using two machine Learning Algorithms:
Local Outlier Factor and Isolation Forest Algorithm. I found that even though I could achieve accuracy of 99.75% for Isolation forest, it is misleading as it falls for False Negatives.

So, as a solution to this, I worked with Logistic Regression. To achieve stability in dataset, I under-sampled it. I used cross-validation for calculating accuracy obtained was about 92.5% and I calculated it using cross-validation technique such as K-fold.
