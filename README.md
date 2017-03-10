# AMATH 582 Project -- Imbalanced Data, Fraud

## Git workflow:
1. git commit -m "fix Taylor's typos"
2. git pull --rebase
3. Resolve any conflicts
4. Git push

## Notes:

### Background
Machine learning algorithms have trouble with unbalanced data. If only a very small proportion of the data has a positive classification then you will see large number of false negatives.

Our dataset has only 492 of 284807 data points that are positive. This means that the positive class represents a mere 0.17% of the total. In this light it's actually extremely easy to build a classifier with 99.83% accuracy--just always predict negative!

However what we care about is, more than anything, not having false negatives. This means that if we have no FNs 99.83% accuracy is the least we can get, but that's not what we expect. It's much more likely that when we optimize for low numbers of FNs we will have many false positives and thus a lower accuracy. However since we value a lower number of FNs so highly we will consider this the better approach. In light of this consideration it's immediately clear that we need a better way to measure success of our algorithm besides accuracy.

#### Performance metric

It's not just accuracy that serves as a poor metric in this task. Precision and Recall are two common measures. Plotting the former on the y-axis and the latter on the x-axis gives you Precision-Recall (PR) curves, which are often used to evaluate algorithms.
Recall is...
Precision is...
The reason these numbers aren't a good metric for this task are...

A metric that does work well is the Area Under the Precision-Recall Curve (AUPRC)...

#### There are 4 common ways to combat this problem.
1. Under-sampling -- Removing data from the larger class to balance the class sizes. May lead to poorer choice to decision line due to losing data at the border of the classes.
2. Oversampling -- Adding extra data points on top of existing point to balance out the class sizes. May lead to overfitting.
3. Synthetic data generation (e.g. SMOTE) -- Generating artificial data from your existing data to generate more information for the classification algorithm to work with. Generated data should stay within the volume object (in n-dim space) that minimally encloses the existing data.
4. Cost functions-- a cost function defines FNs as much less desirable than FPs.

### Classification algorithms
binary classification decision tree - fitctree
binary regression tree - fitrtree

### References and usefulness
https://www3.nd.edu/~dial/publications/dalpozzolo2015calibrating.pdf
https://en.wikipedia.org/wiki/Receiver_operating_characteristic#Area_under_the_curve
