import numpy as np



############### 5A ##############


def generate_W_matrix(n, alpha):
    """
    Generate the weight matrix W for a ring network with n nodes and parameter alpha.

    Args
    ----
    n : Size of the graph
    alpha : Weight parameter 

    
    """


def synchronous_consensus(x0, steps, W):
    """
    Run synchronous consensus on a ring for a given number of iterations.
    Returns an array X of shape (steps+1, n) containing x(0), x(1), ..., x(steps).

    Args
    ----
    x0 : (n,) numpy array for the initial values
    steps: number of iterations
    W : (n,n) weight matrix
    """
    
def randomized_gossip(x0, steps, seed):
    """
    Run randomized gossip on a ring for a given number of iterations.
    Returns an array X of shape (steps+1, n) containing x(0), x(1), ..., x(steps).

    Args
    ----
    x0 : (n,) numpy array for the initial values
    steps: number of iterations
    seed : a random seed for code checking
    """
    np.random.seed(seed)



############### 5B ##############

def sigmoid(z):
    # sigmoid
    pos = z >= 0
    neg = ~pos
    out = np.empty_like(z, dtype=float)
    out[pos] = 1 / (1 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1 + ez)
    return out

def bce_loss(theta, X, y):
    """
    Binary cross-entropy loss averaged over samples.
    y must be in {0,1}.
    """
    z = X @ theta
    loss = - y @ z + np.log1p(np.exp(z)).sum()
    return loss / len(X)
    

def bce_grad(theta, X, y):
    """
    Gradient of the average BCE wrt theta for logistic regression.
    """
    z = X @ theta
    p = sigmoid(z)
    # average gradient
    return (X.T @ (p - y)) / len(X)




def parallel_sgd(X, y, theta0, T, eta, K, batch_size, seed):
    """
    Parallel (synchronous) mini-batch SGD with K workers.

    The data split part is provided for you.

    Args
    ----
    X : (N, d) ndarray
    y : (N,) ndarray in {0,1}
    theta0 : (d,) initial parameter
    T : number of communication rounds (global steps)
    eta : learning rate
    K : number of workers
    batch_size : per-worker mini-batch size
    seed : RNG seed
    
    Returns
    -------
    A tuple of (loss_hist, theta_hist) where
      'loss_hist' : (T+1,) BCE over the full dataset per round
      'theta_hist' : (T+1, d) parameter trajectory
    """
    np.random.seed(seed)
    N, d = X.shape

    # Split the data into K shards 
    shard_idx = np.array_split(np.arange(N), K)
    shard_X = [X[idx] for idx in shard_idx]
    shard_y = [y[idx] for idx in shard_idx]




def local_sgd(X, y, theta0, T, eta, K, batch_size, seed, tau):
    """
    Local SGD with K workers and averaging period `tau`.

    Each communication round r = 1..T:
      - Initialize each worker's local model w_k := current global theta
      - Worker k performs `tau` local SGD steps on its own shard (no communication)
      - Average models
      - Record full-data BCE loss

    Args
    ----
    X : (N, d) ndarray
    y : (N,) ndarray in {0,1}
    theta0 : (d,) initial parameter
    T : number of **communication rounds**
    eta : learning rate
    K : number of workers
    batch_size : per-worker mini-batch size
    seed : RNG seed
    tau : int, number of **local** steps between global averaging

    Returns
    -------
    A tuple of (loss_hist, theta_hist) where
      loss_hist  : (T+1,) BCE over the full dataset per communication round
      theta_hist : (T+1, d) parameter trajectory (after each averaging)
    """
    np.random.seed(seed)
    N, d = X.shape

    # Split the data into K shards 
    shard_idx = np.array_split(np.arange(N), K)
    shard_X = [X[idx] for idx in shard_idx]
    shard_y = [y[idx] for idx in shard_idx]




