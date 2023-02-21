# Stochastic Optimization for Image Classifier
We will consider the following statistical model, defined by a function to minimize. It is called the Multinomial logisitic regression with squared 2-norm regularization. Denote $y_{i,j} = 1$ if the 5 image $i$ represents digit $j$ and 0 otherwise and consider a positive real number $\alpha$. The objective function is the convex function:
$$F(w) = \frac{1}{n} \sum_{i = 1}^{n} F_i(w)$$ 
Where, $\forall i \in \{1, \dots, n\}$:
$$F_i(w) =log~(\sum_{j = 0}^{9} \exp~(\sum_{k = 1}^{d} x_{i,k}w_{k,j} )) - \sum_{j = 0}^{9} y_{i,j}(\sum_{k = 1}^{d} x_{i,k}w_{k,j}) + \frac{\alpha}{2} \||w\||^2_2$$
Here $d$ is the number of pixel in the image xi,: and n is the number of images is the training data set. In this case, you should derive the formula for the stochastic gradients and use this formula in the algorithm.
The work to do is as follows:

* Each student chooses an algorithm to determine the parameters of the model.
* By keeping part of the training set into a validation set, and a good value for the hyper parameter $\alpha$
Let us explain the procedure for the multinomial logistic regression case and forget about $w_0$ for simplication. Denote $\alpha$ the hyperparameter and $A$ the set of its possible values. Let $F_α(w)$ be the loss defined by the statistical model for the parameter $w$ and hyperparameter $\alpha$, that is
$$F_\alpha (w) = \frac{1}{n_{train}} \sum_{i = 1}^{n_{train}} \log~(\sum_{j = 0}^{9} \exp~(\sum_{k = 1}^{d} x_{i,k}w_{k,j}))-\sum_{j = 0}^{9} y_{i,j}(\sum_{k = 1}^{d} x_{i,k}w_{k,j} ) + \frac{\alpha}{2} \||w\||^2_2$$
The optimization problem we are trying to solve in this question is a bilevel optimization problem.
$$max_{\alpha\in A} L(w_\alpha)$$
where $w_\alpha \in argmin_w F_\alpha(w)$ We will then solve a sequence of optimization problems indexed by α and choose the best $\alpha$.
* compare optimization algorithms in terms of classification performance and speed of convergence.
