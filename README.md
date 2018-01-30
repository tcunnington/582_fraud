# AMATH 582 Project -- Imbalanced Data -- Fraud Detection Problem


### Goal of the project:
To consider classification methods in the presence of highly imbalanced data--binary data where a majority class far outnumbers the minority class. We will cover how to evaluate methods in a way that captures the importance of classifying the minority class correctly, as well as ways to improve these methods. 

### Background

The dataset we use for this project is a perfect example of imbalanced data: credit card transactions from european cardholders where some transactions are fraudulent. To a credit card company instances of fraud are relatively rare in comparison to all transactions, but they are particularly important to be able to identify because they are very costly. 

Classifying bank transactions as fraudulent is a difficult task: (1) data is highly sensitive and so difficult to come by, (2) relevant features are not always clear, and (3) classification algorithms have trouble with unbalanced data where fraud is much less common than honest purchases.
To answer the first problem, we analyze real credit card data from a challenge on Kaggle.com. How- ever, to preserve confidentiality, the features have been obscured by a singular value decomposition (SVD): we are given UΣ if the original data $A = UΣV^∗$ in the usual SVD. This means we have no say in the second problem, feature extraction, and so we will not attempt anything in this regard (it doesn't appear to be a good candidate for dimensionlity reduction either). Instead, we focus on classification.
 
 There are two main challenges for us in classifying this data: 
1. Classification algorithms predict the minority class well when trained on such data.
2. The common classifier performance metrics do not align with our preference for predicting the minority class when working with data such as this. This is also the reason that these algorithms do not perform well: they are optimized for a less-ideal metric such as reducing error.

That popular algorithms have trouble with skew is not all that surprising: these algorithms typically work to minimize the error rate, which inherently favors the majority class. Guessing the larger class every time is already above 50% accuracy. Our dataset is very imbalanced: only 0.17% of the data points are classified positive. If we want high accuracy then let's just always guess negative--it's difficult to improve on 99.83% accuracy! We won't be doing that, but we will attempt to alter our methods if possible. We will cover several methods learned in lecture including quadratic discriminant analysis (QDA), a binary classification tree (CART), and logistic regression. 

So clearly overall error is not the metric we care most about. What we _do_ a lot about is false negatives (FNs: someone gets away with fraud). We care to a much lesser extent about false positives (FPs: mistakenly flagging a transaction as fraud). Because our classes are not well separated (see either ipython notebook) we will have to accept a lower accuracy in exchange for prioritizing lower FNs. It should be clear by now that we need to define a performance metric other than accuracy for our task.

#### Performance metric

The baseline metrics for evaluating an imbalanced classification scheme are all contained in the "confusion matrix". Remembering that positive is a fraudulent result, we can construct a matrix which counts the total number of true positive (TP), false positive (FP), true negative (TN), and false negative (FN) labels: C = [TP FN; FP TN]

Using these terms can can define precision: TP/(TP+FP), recall: TP/(TP+FN), accuracy: (TP+TN)/(TP+TN+FP+FN), and more. While this is a convenient construct it does not provide a singlular metric for determining performance. For that we will use recall, or the fraction of positive entries that you correctly identified as positive. We chose this because identifying fraud is far and away the most important goal of our classifiers. Accurancy is pretty much always be high, as explained earlier, and precision is much less important since we assume that FPs are not very costly compared to FNs.

Precision and recall are often considered together in the aptly named Precision-Recall Curve (PRC), or in the related metric: Area Under the Precision-Recall Curve (AUPRC). We will consider this as well as ROC curves.

#### Methods for imbalanced data

There are some known methods for improving performance when classifying imbalanced data. The four common solution types are given here:
1. Under-sampling – Removing data from the larger class to balance the class sizes. May lead to a poorer choice of decision line due to losing data at the border of the classes. Lower number of samples (less information) could also exacerbate overfitting. 
2. Oversampling – Adding additional observations "on top" of existing minority class observations to balance out the class sizes. May lead to overfitting with some classification models since new points are at the exact same locations.
3. Synthetic data generation – Generating artificial data for your minority class from your existing data. Generated data generally stays within the volume that minimally encloses the existing data.
4. Cost functions– Classification algorithms that use cost functions (decision functions) to define their decision boundaries can weigh a FN more heavily than other types of error to encourage the algorithm to essentially prioritize recall.

The first three attempt to reduce the class imbalance by reducing the number of majority data point or increasing the number of minority data points. For synthetic data generation we chose to use the SMOTE (Synthetic Minority Over-Sampling Technique) and ADASYN (Adaptive Synthetic Sampling Approach for Imbalanced Learning) algorithms. SMOTE isis ADASYN’s precursor and ADASYN is essentially a specific implementation of of SMOTE. The goal of both is to improve the class balance by increasing the number of minority class members. SMOTE places synthetic data between existing data points randomly (linear interpolation between neighbors), with no preference shown to any specific points. ADASYN does the same thing but places more synthetic data points close to the boundary between classes because those are the original data points that are more difficult to learn.


### Classification algorithms

##### Gaussian methods

Gaussian methods such as linear discriminant analysis and quadratic discriminant analysis attempt to fit a multivariate normal distribution to existing data. In LDA the covariance matrix is shared by both classes, whereas in QDA they independent. Naive Bayes uses independent covariances but they are constrained to be diagonal. Each class also has a prior probability that determines how likely each class is to be found (calculated as the fraction of one class to the other by default). Due to this prior a rare class will have an extremely small probability of occurring and will not be likely unless the two classes have a very good separation, or the minority class is compact in feature space. If it were you could model it as a sharply peaked MVN that might be able to rise above the advantage the majority class has due to its much larger prior. This immediately suggests that LDA is a poor choice since it forces a shared covariance matrix--this means that the minorty class cannt be sharply peaked relative to the majority class distribution. For this reason we will consider QDA instead. 

##### Decision Trees

Binary decision trees are a common and effective choice for classification tasks. You classify an unknown observation by starting at the root node and at each node in the tree you make binary decision (meaning there are two branches) based on the features of the data used for training. You work your way up the tree until you hit a terminal leaf node. This node determines your prediction. You can think about this process as splitting up m-dimensional space into “blocks” (if you imagine the 3-dimensional version), where m is the number of features and each block represents a class prediction. It starts with the full space and splits it into two, then one of those subspaces in two, and then one of those three subspaces into two, and onwards until it finishes. The algorithm chooses where to put the boundaries based on minimizing the "Gigi index". The Gini index naturally gives high weight to the more represented class, which is a bias for the majority class. One potential issue to keep in mind is overfitting. Trees are known to overfit if you have a high number of features (we have ~30), and since there are relatively few samples of the minority class (492 out of 284807), we may not have enough data on the minority class to make a robust classifier. 

##### Logistic regression

Logistic regression is a method that generates probabilities that a test point is in each class using the logistic regression function. 


### References and usefulness
http://contrib.scikit-learn.org/imbalanced-learn/stable/user_guide.html
https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf
https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/
http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/


## Git workflow:
1. `git commit -m "fix typos"`
2. `git pull --rebase`
3. Resolve any conflicts
4. `git push`