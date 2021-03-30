# Deep Learning


## Neural Network

### A Single Neuron / Perceptron


$$\begin{tikzpicture}[shorten >=1pt,->]
	\tikzstyle{unit}=[draw,shape=circle,minimum size=1.15cm]

	\node[unit](p) at (2,1){$y$};
	\node(dots) at (-0.25,1){\vdots};

	\draw (0,2.5) node[xshift=-10]{$w_0$} -- (p);
	\draw (0,1.75) node[xshift=-10]{$x_1$} --(p);
	\draw (0,0) node[xshift=-10]{$x_D$} -- (p);
	\draw (p) -- (3,1) node[xshift=30]{$y := f(z)$};
\end{tikzpicture}$$

$$z = w x + b$$

adding an activation function g : 

$$z = g(w x + b)$$

Some standard activation functions are : 
* The sigmoid : $g(x) = \dfrac{e^x}{1+e^x}$ 
* The ReLu : $g(x) = \max(0,x)$
* (Can also be linear: $g(x)=x$)
  
  

### Multi Layer Perceptron (MLP)


### Backpropagation 



## Auto-Encoder (AE)

## Variational Auto Encoder (VAE)

## Generative Adversarial Network (GAN)