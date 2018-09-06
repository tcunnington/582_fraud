from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, average_precision_score, make_scorer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import cross_val_score, cross_validate, KFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#################################################
#
#     Utils
#
#################################################

SCORING = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score),
        'recall': make_scorer(recall_score),
        'f1': make_scorer(f1_score),
        'geometric': make_scorer(geometric_mean_score),
    }

def split_class(df):
    y = df['Class']
    x = df.drop(['Class'], axis=1)
    return x, y


def test_model(model, X, y, cv=5, verbose=True):
    scores = cross_validate(model, X, y, scoring=SCORING, cv=cv)

    if verbose:
        print_scores(model, scores)

    return scores

def cv_sampler(sampler, model, X, y, cv=3, verbose=True, **kwargs):
    """
    Run a k-fold cross validation while using a sampling method
    """
    kf = KFold(n_splits=cv)
    scores = defaultdict(list)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train, y_train = sampler.fit_sample(X_train, y_train)
        model.fit(X_train, y_train)

        for name, scorer in SCORING.items():
            scores['test_' + name].append(scorer(model, X_test, y_test))

    # for name, score_list in scores.items():
    #     scores[name] = np.array(score_list)

    scores = {name:np.array(score_list) for name, score_list in scores.items()}

    if verbose:
        print_scores(model, scores)

    return scores

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
    return str(instance).replace("[", "").split("(")[0]

def print_class_counts(y_train):
    print("Class counts:")
    print(sorted(Counter(y_train).items()))

def print_scores(model, scores): # like those from cross_validate
    print('---' + get_class_name(model) + '---')
    print('Precision: \t{0:0.2f}'.format(scores['test_precision'].mean()))
    print('Recall: \t{0:0.2f}'.format(scores['test_recall'].mean()))
    print('F1: \t\t{0:0.2f}'.format(scores['test_f1'].mean()))
    print('Geometric mean: {0:0.2f}'.format(scores['test_geometric'].mean()))

def print_metrics(true, pred, *args):
    print('Classification Report:')
    print(classification_report(true, pred))
    print("Geometric mean:\t", geometric_mean_score(true, pred)) # sqrt(recall * specificity), or sqrt(TPR * TNR))
    # print("Confusion mat: \n", confusion_matrix(true, pred))
    
    for a in args:
        print(a)

def print_avg_recall(model, X, y, *args):
    scores = cross_val_score(model, X, y, scoring='recall', cv=5)
    print('Mean recall score ({0}): {1:0.2f}'.format(get_class_name(model), scores.mean()))
    for a in args:
        print(a)

def print_avg_f1(model, X, y, *args):
    scores = cross_val_score(model, X, y, scoring='f1', cv=5)
    print('Mean f1 score ({0}): {1:0.2f}'.format(get_class_name(model), scores.mean()))

def print_avg_geometric_mean(model, X, y, *args):
    scores = cross_val_score(model, X, y, scoring=make_scorer(geometric_mean_score), cv=5)
    print('Mean geomtric mean score ({0}): {1:0.2f}'.format(get_class_name(model), scores.mean()))


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

    ax.scatter(legit[f1], legit[f2], s=10)
    ax.scatter(fradulent[f1], fradulent[f2], s=10)
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
    fig, axarr = plt.subplots(2, 2, figsize=(11, 8))
    for ax in np.array(axarr).flatten():
        plot_rand_features(feature_list, df, ax=ax)

    plt.suptitle('Example 2D feature plots', size=16)
    axarr[0][0].legend(labels=['legitimate', 'fraudulent'], loc = (0, 1.05), ncol=2)
    plt.show()

def plot_pr(model, test_x, test_y, ax=None, **plot_kwargs):
    if not ax:
        plt.figure()
        ax = plt.subplot(111)

    try:
        y_score = model.decision_function(test_x)
    except:
        y_score = model.predict_proba(test_x)[:,1]

    average_precision = average_precision_score(test_y, y_score)
    precision, recall, _ = precision_recall_curve(test_y, y_score)
    pr_auc = auc(recall, precision)

    if 'label' not in plot_kwargs:
        plot_kwargs['label'] = truncate_dotdot(get_class_name(model))

    title = 'Precision-Recall curve'
    if 'title' in plot_kwargs:
        title += ' -- ' + plot_kwargs['title']
        del plot_kwargs['title']

    plot_kwargs['label'] = plot_kwargs['label'] + ', AUC: {0:0.2f}'.format(pr_auc)

    ax.step(recall, precision, **plot_kwargs) #, color='b', alpha=0.2, where='post')
    # plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    ax.set_title(title)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.legend(loc = 'lower left')

    return ax

def plot_roc(model, test_x, test_y, ax=None, **plot_kwargs):
    if ax is None:
        plt.figure()
        ax = plt.subplot(111)

    try:
        y_score = model.decision_function(test_x)
    except:
        y_score = model.predict_proba(test_x)[:,1]

    fpr, tpr, _ = roc_curve(test_y, y_score)
    roc_auc = auc(fpr, tpr)

    if 'label' not in plot_kwargs:
        plot_kwargs['label'] = truncate_dotdot(get_class_name(model))

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

    return ax


def plot_many_roc(models, labels, test_x, test_y):
    # plt.figure(figsize=(8,6))
    # ax = plt.subplot(111)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for m, l in zip(models, labels):
        plot_roc(m, test_x, test_y, ax=ax1, label=l)

    for m, l in zip(models, labels):
        plot_roc(m, test_x, test_y, ax=ax2, label=l)
        ax2.set_xlim([0.0, 0.4])
        ax2.set_ylim([0.8, 1.0])

    plt.suptitle(get_class_name(models[0]), size=16)
    plt.show()
    # return ax1, ax2


def plot_many_pr(models, labels, test_x, test_y, **plot_kwargs):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for m, l in zip(models, labels):
        plot_pr(m, test_x, test_y, ax=ax1, label=l, **plot_kwargs)

    for m, l in zip(models, labels):
        plot_pr(m, test_x, test_y, ax=ax2, label=l, **plot_kwargs)
        ax2.set_xlim([0.8, 1.0])
        ax2.legend(loc = 'upper right')

    plt.suptitle(get_class_name(models[0]), size=16)
    plt.show()
    # return ax1, ax2


def bar_plot(df, subtitle=None, rot=45, legend_loc='lower right'):
    title = 'Performance metrics'

    if 'subtitle' is not None:
        title += ' -- ' + subtitle

    df.plot.bar(figsize=(10, 6), rot=rot)
    plt.title(title)
    plt.legend(loc=legend_loc)
    plt.yticks(np.arange(0, 11) / 10.0)
    plt.grid(True)
    plt.show()
