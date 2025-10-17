import numpy as np



## some helper functions
def bce_loss(theta, X, y):
    """
    Binary cross-entropy loss averaged over samples.
    """

    

def bce_grad(theta, X, y):
    """
    Gradient of the binary cross-entropy loss function average over samples.
    """


## functions for gradient descent 
def batch_gradient_descent(X, y, gradient_func, theta0, T, eta):
    """
    Full-batch Gradient Descent (GD).

    Parameters
    ----------
    X : an array of size shape (n, d), each row of which is a feature vector

    y : an array of size n, each entry of which is a binary label in {0,1} corresponding to the
        Targets/labels.

    gradient_func : 
        A function that returns the gradient of the training loss
        at the given parameter vector. Signature:
            gradient_func(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray

    theta0 : an arraty of size d
        Initial parameter vector at iteration t=0.

    T : int
        Number of updates. 

    eta : float
        Learning rate (step size). 

    Returns
    -------
    List[np.ndarray]
        Sequence of parameters per epoch, including the initial vector:
        [theta_(0), theta_(1), ..., theta_(T)].
        (Length T+1)
    """



def gradient_descent_with_momentum(X, y, gradient_func, theta0, T, eta, beta ):
    """
    Heavy-ball (Gradient Descent with Momentum).

    Parameters
    ----------
    X : an array of size shape (n, d), each row of which is a feature vector

    y : an array of size n, each entry of which is a binary label in {0,1} corresponding to the
        Targets/labels.

    gradient_func : 
        A function that returns the gradient of the training loss
        at the given parameter vector. Signature:
            gradient_func(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray

    theta0 : an arraty of size d
        Initial parameter vector at iteration t=0.

    T : int
        Number of GD updates. 

    eta : float
        Learning rate (step size). 

    beta : float
        Momentum parameter 

    Returns
    -------
    List[np.ndarray]
        Sequence of parameters per epoch, including the initial vector:
        [theta_(0), theta_(1), ..., theta_(T)].
        (Length T+1)
    
    When you update the first iteration, you can assume theta_(0) = theta_(-1)
    """



def stochastic_gradient_descent(X, y, loss_func, gradient_func, theta0, T, eta, B):
    """
    Stochastic Gradient Descent (SGD) with mini-batches of size B.

    Parameters
    ----------
    X : an array of size shape (n, d), each row of which is a feature vector

    y : an array of size n, each entry of which is a binary label in {0,1} corresponding to the
        Targets/labels.

    gradient_func : 
        A function that returns the gradient of the training loss
        at the given parameter vector. Signature:
            gradient_func(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray

    theta0 : an arraty of size d
        Initial parameter vector at iteration t=0.

    T : int
        Number of GD updates. 

    eta : float
        Learning rate (step size). 

    B : int
        Mini-batch size.


    Returns
    -------
    List[np.ndarray]
        Sequence of parameters per epoch, including the initial vector:
        [theta_(0), theta_(1), ..., theta_(T)].
        (Length T+1)
    
    """



