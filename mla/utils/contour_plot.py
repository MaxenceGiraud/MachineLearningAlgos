import numpy as np
import matplotlib.pyplot as plt


def plot_contour(X,y,model,model_fun='auto',ax=None,precision = 100,decision_boundary=0.5):
    ax = ax if ax is not None else plt.gca()

    assert X.shape[1] == 2, "Data must be in 2D"

    posx = (min(X[:,0]),max(X[:,0]))
    xl = (posx[1]-posx[0])/6
    posy = (min(X[:,1]),max(X[:,1]))
    yl = (posy[1]-posy[0])/6

    xx = np.linspace(posx[0]-xl,posx[1]+xl,precision)
    yy = np.linspace(posy[0]-yl,posy[1]+yl,precision)
    xxx,yyy = np.meshgrid(xx,yy)
    X_grid = np.array([xxx.flatten(),yyy.flatten()]).T

    if model_fun == 'auto' :
        if hasattr(model,'predict_probs'):
            predict = getattr(model,'predict_probs')
        elif hasattr(model,'predict'):
            predict = getattr(model,'predict')
        else :
            raise EnvironmentError('Mode has not known prediction method, specify it manually witht the param model_fun')
    else :
        predict = getattr(model,model_fun)
        
    y_grid = predict(X_grid).reshape(precision,precision)

    if decision_boundary is not None :
        if isinstance(decision_boundary,list):
            pass
        else :
            ax.contour(xxx,yyy,np.where(y_grid>decision_boundary,1,0).reshape(precision,precision))
    contour = ax.contourf(xxx, yyy,y_grid , cmap=plt.cm.bwr, alpha=.8)
    ax.scatter(X[:,0],X[:,1],c=y)
    plt.colorbar(contour,ax=ax)