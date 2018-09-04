# Fraud Detection: An Imbalanced Classification Problem

### The project:

Fraud data is a common example of highly imbalanced data--data where a majority class far outnumbers the minority class--due to the relative rarity of fraud. Finding a high proportion of the minority class can be a challenge for many classifiers. This project is meant to explore specific methods for improving model performance using logistic regression, support vector classifiers, and random forests. For detailed discussion of performance metrics, algorithmic considerations, and sampling methods see the sections below Results.

### Overview

The dataset we use for this project is a perfect example of imbalanced data: credit card transactions from European cardholders where some transactions are fraudulent. To a credit card company instances of fraud are relatively rare in comparison to all transactions, but they are particularly important to be able to identify because they are very costly. This data comes from a challenge on Kaggle.com (https://www.kaggle.com/mlg-ulb/creditcardfraud).

One important feature of this dataset is that the features have been obscured by principle component anaylsis (PCA) to preserve confidentiality. Since we are given just the principle components of the original data we have no way to address feature extraction. Instead, we focus entirely on classification algorithms.

There are two main challenges for us in building a classifier for this data:
1. Many classification algorithms don't predict the minority class well when trained on such data.
2. Some common classifier performance metrics do not align with our preference for predicting the minority class when working with data such as this.

The python library [imbalanced-learn](http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html) was extremely useful in this project. It provided the sampling methods and an implementation of the geometric mean. If you want to see the details of the results, please see the "Imbalanced Data" notebook. If you want to see the initial model selection process where I chose the 3 classifiers to test, please look in "Testing Classifiers".

### Results

For our results we will show currently just the performance of our logistic regression classifier. In the future we will use classification logistic regression, random forests, and SVMs. 
Our primary metric is the geometric mean of the TPR (recall) and the TNR (see Performance Metrics section).
The bar plots below shows that all methods did much better than the baseline on this metric.

![ROC curve](img/geometric_mean_compare.png?raw=true)

Here "balanced" refers to the default model class weighting scheme built into sklearn: the class weight is inversely proportional to it's relative representation in the training data. See the sklearn docs for more details. ADASYN, SMOTE, and balanced do best generally. The test was done with a 3-fold cross validation.

Next we have the precision-recall and ROC curves.

![Precision-recall](img/pr_compare.png?raw=true)

SMOTE and ADASYN have the highest AUC scores, although many seem to perform similarly in the high-recall/low precision region which we would certainly choose to operate in.
Class balanced model and oversampling follow.

![ROC curve](img/roc_compare.png?raw=true)

Again all the methods solidly outperform the baseline, although for the most part their performance is hard to distinguish from each other here, and all have similar AUC values.

Overall we can easily conclude that all of the proposed methods are improvements on the baseline.
The SMOTE, ADASYN and balanced cases performs well in all plots. Balanced had a slightly better geometric mean, which is our primary metric. Oversampling and undersampling appear to be inferior to other methods. Undersampling seems like the weakest according to the ROC and PR curves--perhaps this is explained by the fact that you are losing information to work with.
Based on these results, if I were using logistic regression I would likely just use the balanced class weights since it is the simplest to implement.
If I were using other classification methods I would always compare with SMOTE (it runs faster than ADASYN) if I suspected that imbalanced classes could be a problem, or if I cared about recall more than accuracy.

### Performance metrics

That popular algorithms have trouble with skew is not all that surprising. Algorithms often work to minimize the overall error rate, with false positives and false negatives having the value implicit value. This inherently favors the majority class however. As an example, think about what happens if you simply classified everything as the majority class. Our dataset is very imbalanced: only 0.17% of the data points are classified as the minority class. This means that if we always guess the majority class we will get 99.83% accuracy! This makes it clear that accuracy is not the metric we are most concerned with here. What we actually care _much_ more about is reducing false negatives (FNs: someone gets away with fraud). Of course, we still care about false positives (FPs: mistakenly flagging a transaction as fraud), but to a much lesser extent. With regard to fradualent transactions it's easy to imagine that FNs could be potentially much more expensive to a bank than FPs. Generally speaking, if you are _interested_ in detecting a very small number of anomalous events this will be true. More advantageous metrics than accuracy are recall, the F1 score, or the geometric mean of recall (sensitivity) and the specificity.

The baseline metrics for evaluating a classification scheme are often built from elements of the "confusion matrix". Remembering that positive is a fraudulent result, we can construct a matrix which counts the total number of true positive (TP), false positive (FP), true negative (TN), and false negative (FN) labels: C = [TP FN; FP TN]. Using these terms we can define precision: TP/(TP+FP), recall (sensitivity): TP/(TP+FN), accuracy: (TP+TN)/(TP+TN+FP+FN), specificity: TN/(TN+FP), and more. 

As we stated we will use recall, or the fraction of positive entries that you correctly identified as positive as our primary metric. However because our classes are not well separated (see iPython notebooks) we will have to expect some harsh trade offs between precision and recall. This suggests we use a composite metric that encompasses our secondary goal as well: to reduce the overall false positve rate. Maximizing precision or specificity would help us work towards this goal. To capture these metrics we will use two composite metrics in our model selection analysis: the F1 score (the [harmonic mean](https://en.wikipedia.org/wiki/F1_score) of precision and recall) and the [geometric mean](https://en.wikipedia.org/wiki/Geometric_mean) of recall and specificity.

Precision and recall are often considered together in the aptly named Precision-Recall Curve (PRC), or in the related metric: Area Under the Precision-Recall Curve (AUPRC). We will consider this as well as ROC curves.

### Methods for imbalanced data

There are some known methods for improving performance when classifying imbalanced data. The four approaches are:
1. Under-sampling – Removing data from the larger class to balance the class sizes. May lead to a poorer choice of decision line due to having less information, however could help reduce compute time.
2. Oversampling – Adding additional observations by duplicating existing minority class observations. Likely to lead to overfitting with some classification models since new points are at the exact same locations.
3. Synthetic data generation – Generating artificial data for your minority class based on your existing data. Generated data generally stays within the volume that minimally encloses the existing data.
4. Algorithm modifications – Classification algorithms always use some sort of decision function or cost function to learn their decision boundaries. Depending on the algorithm it's possible to encourage it to essentially prioritize recall, or to be more skew insensitive.

The first three methods are sampling technques, meaning they attempt to reduce the class imbalance by reducing the number of majority data points or increasing the number of minority data points. The synthetic data generation techniques we chose to use were the SMOTE (Synthetic Minority Over-Sampling Technique) and ADASYN (Adaptive Synthetic Sampling Approach for Imbalanced Learning) algorithms. SMOTE places synthetic data between existing data points randomly (using linear interpolation between neighbors), with no preference shown to any specific points. ADASYN is essentially a specific implementation of SMOTE. The only difference is that it places more of the synthetic data points close to the boundary between classes. The idea here is that those are the original data points that are most difficult to learn.

##### Decision Trees

Each split on a particular feature is chosen based on minimizing the Gini impurity. However the Gini impurity measure is skew sensitive and is biased towards the majority class. The other measure used by sci-kit learn, information gain, is less sensitive, but is not insensitive. For this reason, if using either of these decision functions it's suggested to also use a sampling technique.

##### Logistic Regression

The loss function used for this algorithm is the cross entropy, also known as log loss. This function sums loss due to each point individually, so it can be reformulated to weight the incorrectly labelled minority points more highly than the incorrect majority points. Scikit already implements this alteration.

##### SVM

The cost function of the support vector classifier will include incorrectly labelled points as support vectors. A cost values is associated with each of these. We can weight the FNs more highly than the FPs.



### References and usefulness
http://contrib.scikit-learn.org/imbalanced-learn/stable/index.html
https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf
https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/
http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
https://pdfs.semanticscholar.org/40b9/613a3f48ca98bff5ff243dd15cd9200d1ae4.pdf
