import numpy as np
import pandas as pd
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os

def extract_notes(path, naming_conv, notes, ds_type="train", label="pos"):
    """ Extract clinical notes and store each in its own file. This makes it easier to work with fast.ai"""
    
    compiled_path = path + ds_type + "/"
    # Check if base path exists
    if(not os.path.exists(compiled_path)):
        os.mkdir(compiled_path)
    
    # add ds_type folder
    # Check if labeled directory exists
    if(label != "None"):
        compiled_path = path + ds_type + "/" + label + "/"
        if(not os.path.exists(compiled_path)):
            os.mkdir(compiled_path)
    
    print("Extracting", len(notes), "notes to", compiled_path)
    for idx in notes.index:
        n = notes.loc[idx].TEXT
        with open("{base}{fname}_{id}.txt".format(base=compiled_path, fname=naming_conv, id=idx), "w") as f:
            f.write(n)
    
    assert len(os.listdir(compiled_path)) == len(notes), 'Not all the notes were successfully extracted.'
    
def f1_precision_recall(inp, preds):
    """Calculate the F1 Score, precision, and recall for the training, validation, or test set."""
    
    f1 = metrics.f1_score(inp, preds)
    pre = metrics.precision_score(inp, preds)
    recall = metrics.recall_score(inp, preds)
    
    return (f1, pre, recall)

def auc(inp, preds):
    """Calculate the AUC for the training, validation, or test set."""
    
    fpr, tpr, _ = metrics.roc_curve(inp, preds)
    return metrics.auc(fpr, tpr)

def plot_roc(meta):
    """Plot the ROC curve for the training, validation, and test set """
    
    train_fpr, train_tpr, _ = metrics.roc_curve(meta["train_act"], meta["train_preds"])
    valid_fpr, valid_tpr, _ = metrics.roc_curve(meta["valid_act"], meta["valid_preds"])
    test_fpr, test_tpr, _ = metrics.roc_curve(meta["test_act"], meta["test_preds"])
    
    train_auc = metrics.auc(train_fpr, train_tpr)
    valid_auc = metrics.auc(valid_fpr, valid_tpr)
    test_auc = metrics.auc(test_fpr, test_tpr)
    
    plt.figure()
    lw = 2
    
    plt.plot(train_fpr, train_tpr, color='darkorange',
         lw=lw, label='Train ROC curve (area = %0.2f)' % train_auc)
    
    plt.plot(valid_fpr, valid_tpr, color='darkgreen',
         lw=lw, label='Valid ROC curve (area = %0.2f)' % valid_auc)
    
    plt.plot(test_fpr, test_tpr, color='darkred',
         lw=lw, label='Test ROC curve (area = %0.2f)' % test_auc)
    
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


def J_statistic(inp, preds):
    """ Calculate Youden's J-Statistic to determine optimial maximum threshold"""
    fpr, tpr, thresh = metrics.roc_curve(inp, preds)
    idx = np.argmax(tpr - fpr) 
    best_thresh = thresh[idx]
    return best_thresh

def pos_accuracy(inp, preds, thresh=0.5):
    """ Evaluate predictions based on threshold and compute accuracy for positive class."""
    thresh_preds = np.array([1 if pred > thresh and inp[i] == 1 else 0 for i, pred in enumerate(preds)])
    pos_acc = thresh_preds.sum()/inp[inp == 1].sum()
    return (pos_acc)
