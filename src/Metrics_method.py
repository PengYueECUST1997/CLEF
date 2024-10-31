from sklearn import metrics
import numpy as np


def compute_acc(all_yhat, all_labels, flexible_cutoff = False):
    if flexible_cutoff:
        best_threshold = 0
        highest_score = 0
        for threshold in np.arange(0.1, 0.9, 0.01):  
          y_pred = (np.array(all_yhat) > threshold).astype(int)
          score = metrics.accuracy_score(np.array(all_labels), y_pred)
          if score > highest_score:
              highest_score = score
              best_threshold = threshold
    else:
        threshold = 0.5
        y_pred = (np.array(all_yhat) > threshold).astype(int)
        highest_score = metrics.accuracy_score(np.array(all_labels), y_pred)
    return highest_score
  
  
def compute_f1(all_yhat, all_labels, flexible_cutoff = False):
    if flexible_cutoff:
        best_threshold = 0
        highest_score = 0
        for threshold in np.arange(0.1, 0.9, 0.01):  
          y_pred = (np.array(all_yhat) > threshold).astype(int)
          score = metrics.f1_score(np.array(all_labels), y_pred)
          if score > highest_score:
              highest_score = score
              best_threshold = threshold
    else:
        threshold = 0.5
        y_pred = (np.array(all_yhat) > threshold).astype(int)
        highest_score = metrics.f1_score(np.array(all_labels), y_pred)
    return highest_score


def compute_mcc(all_yhat, all_labels, flexible_cutoff = False):
    if flexible_cutoff:
        best_threshold = 0
        highest_score = 0
        for threshold in np.arange(0.1, 0.9, 0.01):  
          y_pred = (np.array(all_yhat) > threshold).astype(int)
          score = metrics.matthews_corrcoef(np.array(all_labels), y_pred)
          if score > highest_score:
              highest_score = score
              best_threshold = threshold
    else:
        threshold = 0.5
        y_pred = (np.array(all_yhat) > threshold).astype(int)
        highest_score = metrics.matthews_corrcoef(np.array(all_labels), y_pred)
    return highest_score

def compute_auc(all_yhat, all_labels, flexible_cutoff = False):
    highest_score = metrics.roc_auc_score(np.array(all_labels), np.array(all_yhat))
    return highest_score
  
def compute_auprc(all_yhat, all_labels, flexible_cutoff = False):
    highest_score =  metrics.average_precision_score(np.array(all_labels), np.array(all_yhat))
    return highest_score

def compute_sn(all_yhat, all_labels, flexible_cutoff = False):
    if flexible_cutoff:
        best_threshold = 0
        highest_score = 0
        for threshold in np.arange(0.1, 0.9, 0.01):
          y_pred = (np.array(all_yhat) > threshold).astype(int)
          tn, fp, fn, tp = metrics.confusion_matrix(np.array(all_labels), y_pred).ravel()
          score = tp / (tp + fn)
          if score > highest_score:
              highest_score = score
              best_threshold = threshold
    else:
        threshold = 0.5
        y_pred = (np.array(all_yhat) > threshold).astype(int)
        tn, fp, fn, tp = metrics.confusion_matrix(np.array(all_labels), y_pred).ravel()
        highest_score =  tp / (tp + fn)
    return highest_score
  
def compute_sp(all_yhat, all_labels, flexible_cutoff = False):
    if flexible_cutoff:
        best_threshold = 0
        highest_score = 0
        for threshold in np.arange(0.1, 0.9, 0.01):
          y_pred = (np.array(all_yhat) > threshold).astype(int)
          tn, fp, fn, tp = metrics.confusion_matrix(np.array(all_labels), y_pred).ravel()
          score = tn / (tn + fp)
          if score > highest_score:
              highest_score = score
              best_threshold = threshold
    else:
        threshold = 0.5
        y_pred = (np.array(all_yhat) > threshold).astype(int)
        tn, fp, fn, tp = metrics.confusion_matrix(np.array(all_labels), y_pred).ravel()
        highest_score =  tn / (tn + fp)
    return highest_score

def compute_pr(all_yhat, all_labels, flexible_cutoff = False):
    if flexible_cutoff:
        best_threshold = 0
        highest_score = 0
        for threshold in np.arange(0.1, 0.9, 0.01):
          y_pred = (np.array(all_yhat) > threshold).astype(int)
          score = metrics.precision_score(np.array(all_labels), y_pred)
          if score > highest_score:
              highest_score = score
              best_threshold = threshold
    else:
        threshold = 0.5
        y_pred = (np.array(all_yhat) > threshold).astype(int)
        highest_score =  metrics.precision_score(np.array(all_labels), y_pred)
    return highest_score


def compute_acc_multi(all_yhat, all_labels, flexible_cutoff = False):
    y_pred = np.array([x.argmax() for x in all_yhat]).astype(int)
    highest_score = metrics.accuracy_score(np.array(all_labels), y_pred)
    return highest_score
  
def compute_acc_multi_binary(all_yhat, all_labels, flexible_cutoff = False):
    if flexible_cutoff:
        best_threshold = 0
        highest_score = 0
        for threshold in np.arange(0.1, 0.9, 0.01):  
          y_pred = (np.array(all_yhat) > threshold).astype(int).flatten()
          y_labels = np.array(all_labels).flatten()
          score = metrics.accuracy_score(y_labels, y_pred)
          if score > highest_score:
              highest_score = score
              best_threshold = threshold
    else:
        threshold = 0.5
        y_pred = (np.array(all_yhat) > threshold).astype(int).flatten()
        y_labels = np.array(all_labels).flatten()
        highest_score = metrics.accuracy_score(y_labels, y_pred)
    return highest_score


def compute_macro_f1(all_yhat, all_labels, flexible_cutoff = False):
    y_pred = np.argmax(np.array(all_yhat), axis=1)

    highest_score = metrics.f1_score(all_labels, y_pred, average='macro')
    
    return highest_score
  

def compute_macro_mcc(all_yhat, all_labels, flexible_cutoff = False):

    y_pred = np.argmax(np.array(all_yhat), axis=1)
    num_classes = all_yhat.shape[1]
    mcc_scores = []
    for cls in range(num_classes):
        binarized_labels = (all_labels == cls).astype(int)
        binarized_predictions = (y_pred == cls).astype(int)
        
        mcc = metrics.matthews_corrcoef(binarized_labels, binarized_predictions)
        mcc_scores.append(mcc)

    highest_score = np.mean(mcc_scores)
    
    return highest_score
