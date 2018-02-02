from collections import Counter
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, average_precision_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#################################################
#
#     Utils
#
#################################################

def split_class(df):
    y = df['Class']
    x = df.drop(['Class'], axis=1)
    return x, y


def run_model(Model,train_x,train_y,test_x,test_y,**kwargs):
    model = Model(**kwargs)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)

    print '---' + get_class_name(model) + '---'
    print_metrics(test_y, pred)

    return model

def run_model_sampler(Sampler,Model,train_x,train_y,test_x,test_y,**model_kwargs):
    """Wrapper for run_model to include sampling"""
    train_x_res, train_y_res = Sampler().fit_sample(train_x, train_y)
    print "Sampler: " + get_class_name(Sampler())
    return run_model(Model, train_x_res, train_y_res, test_x, test_y, **model_kwargs)


#################################################
#
#     Prints
#
#################################################

def truncate_dotdot(str):
    if len(str) > 15:
        str = str[0:13] + '...'
    return str

def get_class_name(instance):
    return str(instance).split("(")[0]

def print_class_counts(y_train):
    print "Class counts:"
    print(sorted(Counter(y_train).items()))

def print_metrics(true, pred, *args):
    print 'Classification Report:'
    print classification_report(true, pred)
    print "Geometric mean:\t", geometric_mean_score(true, pred) # sqrt(recall * specificity), or sqrt(TPR * TNR)
    print "Confusion mat: \n", confusion_matrix(true, pred)
    for a in args:
        print a

def print_avg_recall(model, X, y, *args):
    scores = cross_val_score(model, X, y, scoring='recall', cv=5)
    print 'Mean recall score ({0}): {1:0.2f}'.format(get_class_name(model), scores.mean())
    for a in args:
        print a

#################################################
#
#     Plot helpers
#
#################################################


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

def plot_pr(model, test_x, test_y, ax=None, **plot_kwargs):
    if not ax:
        plt.figure()
        ax = plt.subplot(111)

    y_score = model.decision_function(test_x)

    average_precision = average_precision_score(test_y, y_score)
    precision, recall, _ = precision_recall_curve(test_y, y_score)
    pr_auc = auc(recall, precision)
    model_name = truncate_dotdot(get_class_name(model))


    if 'label' not in plot_kwargs:
        plot_kwargs['label'] = model_name

    plot_kwargs['label'] = plot_kwargs['label'] + ', AUC: {0:0.2f}'.format(pr_auc)

    plt.step(recall, precision, **plot_kwargs) #, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    ax.legend(loc = 'lower left')
    ax.set_title('Precision-Recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])

    return ax

def plot_roc(model, test_x, test_y, ax=None, **plot_kwargs):
    if not ax:
        plt.figure()
        ax = plt.subplot(111)

    y_score = model.decision_function(test_x)
    fpr, tpr, _ = roc_curve(test_y, y_score)
    roc_auc = auc(fpr, tpr)
    model_name = truncate_dotdot(get_class_name(model))

    if 'label' not in plot_kwargs:
        plot_kwargs['label'] = model_name

    plot_kwargs['label'] = plot_kwargs['label'] + ', AUC: {0:0.2f}'.format(roc_auc)

    lw = 2
    ax.plot(fpr, tpr, lw=lw, **plot_kwargs)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')

    ax.legend(loc="lower right")


def plot_many_roc(models, labels, test_x, test_y):
    plt.figure(figsize=(8,6))
    ax = plt.subplot(111)

    for m, l in zip(models, labels):
        plot_roc(m, test_x, test_y, ax=ax, label=l)

    plt.show()


def plot_many_pr(models, labels, test_x, test_y):
    plt.figure(figsize=(8,6))
    ax = plt.subplot(111)

    for m, l in zip(models, labels):
        plot_pr(m, test_x, test_y, ax=ax, label=l)

    plt.show()
