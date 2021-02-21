import numpy as np

def confusion_matrix(y_true,y_pred,normalize=False):
    true_unique = np.unique(y_true)
    pred_unique = np.unique(y_pred)

    conf = np.array([np.sum(y_pred[np.where(y_true==i)[0]] == j) for j in true_unique for i in pred_unique]).reshape(true_unique.size,pred_unique.size)

    if normalize :
        conf =  conf / conf.sum(axis=1).reshape(-1,1)

    return conf