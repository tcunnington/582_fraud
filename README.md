# AMATH 582 Project -- Imbalanced Data, Fraud

## Git workflow:
1. `git commit -m "fix Taylor's typos"``
2. `git pull --rebase`
3. Resolve any conflicts
4. `git push`

## Work to do:
1. Define scope of project

## Ideas on next steps:
Sounds like synthetic data generation performs better that over/under sampling. I'd prefer to try that or cost function methods. Over/under sampling seem hacky and kinda stupid.

Or should we do them all to compare??

The top kernel on Kaggle takes this approach:
We are not going to perform feature engineering in first instance. The dataset has been downgraded in order to contain 30 features (28 anonymized + time + amount).
We will then compare what happens when using resampling and when not using it. We will test this approach using a simple logistic regression classifier.
We will evaluate the models by using some of the performance metrics mentioned above.
We will repeat the best resampling/not resampling method, by tuning the parameters in the logistic regression classifier.
We will finally perform classifications model using other classification algorithms.

## Notes:

### Background
Machine learning algorithms have trouble with unbalanced data. If only a very small proportion of the data has a positive classification then you will see large number of false negatives.

Our dataset is fraud data...

Since the vast majority of the data have been anonymized we will not be performing and feature engineering/analysis. We will instead concentrate on methods to deal with imbalanced data...

Our dataset has only 492 of 284807 data points that are positive. This means that the positive class represents a mere 0.17% of the total. In this light it's actually extremely easy to build a classifier with 99.83% accuracy--just always predict negative!

However what we care about is, more than anything, not having false negatives. This means that if we have no FNs 99.83% accuracy is the least we can get, but that's not what we expect. It's much more likely that when we optimize for low numbers of FNs we will have many false positives and thus a lower accuracy. However since we value a lower number of FNs so highly we will consider this the better approach. In light of this consideration it's immediately clear that we need a better way to measure success of our algorithm besides accuracy.

#### Performance metric

It's not just accuracy that serves as a poor metric in this task. Precision and Recall are two common measures. Plotting the former on the y-axis and the latter on the x-axis gives you Precision-Recall (PR) curves, which are often used to evaluate algorithms.
Recall is the fraction of positive entries that you correctly identified as positive: tp/(tp+fn).
Precision is the fraction of entries that were actually positive out of all the ones you guessed were positive: tp/(tp+fp).

In our case we want to optimize for recall over precision. That is, when we are given an input that is actually positive we want to correctly predict as much as possible that it is positive, even if that means we end up with false positives. If we get false positives that lowers our precision, but since false negatives are more costly than false positives here we can accept potentially many more false positives.  

For the reason above precision is not a good metric for this task. For the functionality we want to optimize we will likely see precision increase over a simple majority class classifier, but we can't really be sure of the exact relationship. A better metric would be one that is guaranteed to decrease monotonically as we approach our ideal of perfect recall, regardless of what happens to precision.

A metric that does work well is the Area Under the Precision-Recall Curve (AUPRC)...

#### There are 4 common ways to combat this problem.
1. Under-sampling -- Removing data from the larger class to balance the class sizes. May lead to poorer choice to decision line due to losing data at the border of the classes.
2. Oversampling -- Adding extra data points on top of existing point to balance out the class sizes. May lead to overfitting.
3. Synthetic data generation (e.g. SMOTE) -- Generating artificial data from your existing data to generate more information for the classification algorithm to work with. Generated data should stay within the volume object (in n-dim space) that minimally encloses the existing data.
4. Cost functions-- a cost function defines FNs as much less desirable than FPs.



### Classification algorithms
binary classification decision tree - `fitctree`
binary regression tree - `fitrtree`
These are CART algorithms. These are binary decision trees. At each node in the tree you can a binary decision and you work your way down the tree until you hit a terminal leaf--this is your determination. You can think about this process as splitting up m-dimensional space into blocks, where m is the number of features. You first divide the entire space into two, then one of those subspaces in two, and onwards again and again until you have made a decision based on all the features. Each of the resulting (m+1?) blocks represents a subspace that corresponds to s specific determination: a class.

The question then becomes how to build the tree. What are the split points chosen to create the best classifier? This is accomplished by minimizing a cost function for training points in a given block you are trying to split. The algorithm is greedy, meaning it always chooses the best prediction that each split point; it does not consider error globally. For regression predictive modeling CART (likely) minimizes the squared error. For classification it uses the Gini cost function.




### References and usefulness
https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf
https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
http://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/
