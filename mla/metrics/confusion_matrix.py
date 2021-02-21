import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(y_true,y_pred,normalize=False):
    true_unique = np.unique(y_true)
    pred_unique = np.unique(y_pred)

    conf = np.array([np.sum(y_pred[np.where(y_true==i)[0]] == j) for j in true_unique for i in pred_unique]).reshape(true_unique.size,pred_unique.size)

    if normalize :
        conf =  conf / conf.sum(axis=1).reshape(-1,1)

    return conf


def plot_confusion_matrix(y_true,y_pred,normalize=False,ax=None,label_names = None,cmap="YlGn"):
    ax = ax if ax is not None else plt.gca()

    conf = confusion_matrix(y_true=y_true,y_pred=y_pred,normalize=normalize)

    im = ax.imshow(conf,cmap=cmap)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    if label_names is not None :
        # We want to show all ticks...
        ax.set_xticks(np.arange(len(label_names)))
        ax.set_yticks(np.arange(len(label_names)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",   rotation_mode="anchor")

    text_colors= ('black','white')
    threshold  = im.norm(conf.max())/ 2

    def truncate(x):
        return int(x* 1e3)/1e3

    # Loop over data dimensions and create text annotations.
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            color=text_colors[int(im.norm(conf[j, i]) > threshold)]
            text = ax.text(i, j, truncate(conf[j, i]),
                        ha="center", va="center", color=color)

    ax.set_xlabel('Prediction')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')

    return ax
