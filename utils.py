from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score, precision_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve, auc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#################################################
#
#     Utils
#
#################################################

def split_by_class(df):
    y = df['Class']
    x = df.drop(['Class'], axis=1)
    return x, y


def run_model(Model,train_x,train_y,test_x,test_y,**kwargs):
    model = Model(**kwargs)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    print get_class_name(model)
    print_metrics(test_y, pred)

    return model


def run_compare_sampling(Sampler,Model,train_x,train_y,test_x,test_y,**model_kwargs):
    print "Sampler: " + get_class_name(Sampler())
    train_x_res, train_y_res = Sampler().fit_sample(train_x, train_y)
    print_class_counts(train_y_res)

    print ''
    run_model(Model, train_x, train_y, test_x, test_y,**model_kwargs)
    print ''
    run_model(Model, train_x_res, train_y_res, test_x, test_y,**model_kwargs)
    print ''


#################################################
#
#     Prints
#
#################################################

def get_class_name(instance):
    return str(instance).split("(")[0]

def print_class_counts(y_train):
    print "Class counts:"
    print(sorted(Counter(y_train).items()))

def print_metrics(true, pred, *args):
    print "Recall: \t", recall_score(true, pred)
    print "Precision:\t", precision_score(true, pred)
    print "F1 Score:\t", f1_score(true, pred)
    print "Accuracy:\t", accuracy_score(true, pred)
    print "Confusion mat: \n", confusion_matrix(true, pred)
    for a in args:
        print a

#################################################
#
#     Plot helpers
#
#################################################


def plot_pr(model, test_x, test_y):
    y_score = model.decision_function(test_x)

    average_precision = average_precision_score(test_y, y_score)
    precision, recall, _ = precision_recall_curve(test_y, y_score)

    # TODO make more pretty?
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()


def plot_features(f1, f2, df, ax=None):
    if not ax:
        plt.figure()
        ax = plt.subplot(111)

    fradulent = df[df['Class'] == 1]
    legit = df[df['Class'] == 0]

    ax.scatter(legit[f1], legit[f2])
    ax.scatter(fradulent[f1], fradulent[f2])
    ax.set_xlabel(f1)
    ax.set_ylabel(f2)
    return ax


def compare_features(*args):
    plot_features(*args)
    plt.show()


def plot_rand_features(feature_list, df, **kwargs):
    cols = list(np.random.choice(feature_list, 2, replace=False))
    return plot_features(*(cols + [df]), **kwargs)


def tile_random_features(feature_list, df):
    fig, axarr = plt.subplots(2, 2, figsize=(13, 10))
    for ax in np.array(axarr).flatten():
        plot_rand_features(feature_list, df, ax=ax)

    plt.show()

def plot_roc(test_y, y_score):
    fpr, tpr, _ = roc_curve(test_y, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
