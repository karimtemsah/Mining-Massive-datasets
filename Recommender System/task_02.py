import scipy.sparse as sp
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
import time


def cold_start_preprocessing(matrix, min_entries):
    """
    Recursively removes rows and columns from the input matrix which have less than min_entries nonzero entries.
    
    Parameters
    ----------
    matrix      : sp.spmatrix, shape [N, D]
                  The input matrix to be preprocessed.
    min_entries : int
                  Minimum number of nonzero elements per row and column.

    Returns
    -------
    matrix      : sp.spmatrix, shape [N', D']
                  The pre-processed matrix, where N' <= N and D' <= D
        
    """
    print("Shape before: {}".format(matrix.shape))
    
    ### YOUR CODE HERE ###
    while True:
        shape = matrix.shape
        matrix = matrix[matrix.getnnz(1)>min_entries]
        matrix = matrix[:,matrix.getnnz(0)>min_entries]
        shape_new = matrix.shape
        if shape == shape_new:
            break
    print("Shape after: {}".format(matrix.shape))
    nnz = matrix>0
    assert (nnz.sum(0).A1 > min_entries).all()
    assert (nnz.sum(1).A1 > min_entries).all()
    return matrix

def shift_user_mean(matrix):
    """
    Subtract the mean rating per user from the non-zero elements in the input matrix.
    
    Parameters
    ----------
    matrix : sp.spmatrix, shape [N, D]
             Input sparse matrix.
    Returns
    -------
    matrix : sp.spmatrix, shape [N, D]
             The modified input matrix.
    
    user_means : np.array, shape [N, 1]
                 The mean rating per user that can be used to recover the absolute ratings from the mean-shifted ones.

    """
    
    ### YOUR CODE HERE ###
    tot = np.array(matrix.sum(axis=1).squeeze())[0]
    cts = np.diff(matrix.indptr)
    user_means = tot/cts
    d = sp.diags(user_means, 0)
    b = matrix.copy()
    b.data = np.ones_like(b.data)
    matrix = (matrix - d*b)   
    assert np.all(np.isclose(matrix.mean(1), 0))
    return matrix, user_means

def split_data(matrix, n_validation, n_test):
    """
    Extract validation and test entries from the input matrix. 
    
    Parameters
    ----------
    matrix          : sp.spmatrix, shape [N, D]
                      The input data matrix.
    n_validation    : int
                      The number of validation entries to extract.
    n_test          : int
                      The number of test entries to extract.

    Returns
    -------
    matrix_split    : sp.spmatrix, shape [N, D]
                      A copy of the input matrix in which the validation and test entries have been set to zero.
    
    val_idx         : tuple, shape [2, n_validation]
                      The indices of the validation entries.
    
    test_idx        : tuple, shape [2, n_test]
                      The indices of the test entries.
    
    val_values      : np.array, shape [n_validation, ]
                      The values of the input matrix at the validation indices.
                      
    test_values     : np.array, shape [n_test, ]
                      The values of the input matrix at the test indices.

    """
    
    ### YOUR CODE HERE ###
    indices = matrix.nonzero()
    a = np.arange(0, len(indices[0]), 1)
    np.random.shuffle(a)
    val_idx = ([],[])
    test_idx = ([],[])
    val_values = np.zeros(n_validation)
    test_values = np.zeros(n_test)
    matrix_split = matrix.copy()


    for i in range(n_validation):
            val_idx[0].append(indices[0][a[i]])
            val_idx[1].append(indices[1][a[i]])
            val_values[i] = matrix[indices[0][a[i]],indices[1][a[i]]]
            matrix_split[indices[0][a[i]],indices[1][a[i]]] = 0
            
            
    for i in range(n_test):
            test_idx[0].append(indices[0][a[i + n_validation]])
            test_idx[1].append(indices[1][a[i + n_validation]])
            test_values[i] = matrix[indices[0][a[i + n_validation]],indices[1][a[i + n_validation]]]
            matrix_split[indices[0][a[i + n_validation]],indices[1][a[i + n_validation]]] = 0
    
    matrix_split.eliminate_zeros()
    return matrix_split, val_idx, test_idx, val_values, test_values

def initialize_Q_P(matrix, k, init='random'):
    """
    Initialize the matrices Q and P for a latent factor model.
    
    Parameters
    ----------
    matrix : sp.spmatrix, shape [N, D]
             The matrix to be factorized.
    k      : int
             The number of latent dimensions.
    init   : str in ['svd', 'random'], default: 'random'
             The initialization strategy. 'svd' means that we use SVD to initialize P and Q, 'random' means we initialize
             the entries in P and Q randomly in the interval [0, 1).

    Returns
    -------
    Q : np.array, shape [N, k]
        The initialized matrix Q of a latent factor model.

    P : np.array, shape [k, D]
        The initialized matrix P of a latent factor model.
    """

    
    if init == 'svd':
        U, sigma, P = svds(matrix, k)
        x = np.eye(k)
        sigma = x * sigma
        Q = np.matmul(U, sigma)
    elif init == 'random':
        Q = np.random.rand(matrix.shape[0], k)
        P = np.random.rand(k, matrix.shape[1])
    else:
        raise ValueError
        
    assert Q.shape == (matrix.shape[0], k)
    assert P.shape == (k, matrix.shape[1])
    return Q, P

def latent_factor_gradient_descent(M, non_zero_idx, k, val_idx, val_values, 
                                   reg_lambda, learning_rate, batch_size=-1,
                                   max_steps=50000, init='random',
                                   log_every=1000, patience=20,
                                   eval_every=50):
    """
    Perform matrix factorization using gradient descent. Training is done via patience,
    i.e. we stop training after we observe no improvement on the validation loss for a certain
    amount of training steps. We then return the best values for Q and P oberved during training.
    
    Parameters
    ----------
    M                 : sp.spmatrix, shape [N, D]
                        The input matrix to be factorized.
                      
    non_zero_idx      : np.array, shape [nnz, 2]
                        The indices of the non-zero entries of the un-shifted matrix to be factorized. 
                        nnz refers to the number of non-zero entries. Note that this may be different
                        from the number of non-zero entries in the input matrix M, e.g. in the case
                        that all ratings by a user have the same value.
    
    k                 : int
                        The latent factor dimension.
    
    val_idx           : tuple, shape [2, n_validation]
                        Tuple of the validation set indices.
                        n_validation refers to the size of the validation set.
                      
    val_values        : np.array, shape [n_validation, ]
                        The values in the validation set.
                      
    reg_lambda        : float
                        The regularization strength.

    learning_rate     : float
                        Step size of the gradient descent updates.
                        
    batch_size        : int, optional, default: -1
                        (Mini-) batch size. -1 means we perform standard full-sweep gradient descent.
                        If the batch size is >0, use mini batches of this given size.
                        
    max_steps         : int, optional, default: 100
                        Maximum number of training steps. Note that we will stop early if we observe
                        no improvement on the validation error for a specified number of steps
                        (see "patience" for details).
                      
    init              : str in ['random', 'svd'], default 'random'
                        The initialization strategy for P and Q. See function initialize_Q_P for details.
    
    log_every         : int, optional, default: 1
                        Log the training status every X iterations.
                    
    patience          : int, optional, default: 10
                        Stop training after we observe no improvement of the validation loss for X evaluation
                        iterations (see eval_every for details). After we stop training, we restore the best 
                        observed values for Q and P (based on the validation loss) and return them.
                      
    eval_every        : int, optional, default: 1
                        Evaluate the training and validation loss every X steps. If we observe no improvement
                        of the validation error, we decrease our patience by 1, else we reset it to *patience*.
                        
    Returns
    -------
    best_Q            : np.array, shape [N, k]
                        Best value for Q (based on validation loss) observed during training
                      
    best_P            : np.array, shape [k, D]
                        Best value for P (based on validation loss) observed during training
                      
    validation_losses : list of floats
                        Validation loss for every evaluation iteration, can be used for plotting the validation
                        loss over time.
                        
    train_losses      : list of floats
                        Training loss for every evaluation iteration, can be used for plotting the training
                        loss over time.                     
    
    converged_after   : int
                        it - patience*eval_every, where it is the iteration in which patience hits 0,
                        or -1 if we hit max_steps before converging. 

    """
    temp_patience = patience
    initial_val_loss = 10000000000
    train_losses = []
    validation_losses = []
    Q, P = initialize_Q_P(M, k, init)
    for step in range(max_steps):
        print(step)
        Q_new = Q.copy()
        P_new = P.copy()
         for count in range(len(non_zero_idx[0])):
	        i = non_zero_idx[0][count]
	        j = non_zero_idx[1][count]
	        eij = M[i,j] - np.dot(Q[i,:], P[:,j])

	        Q_new[i, :] = Q[i, :] + learning_rate * (2 * eij * P[:, j] - reg_lambda * Q[i, :])
	        P_new[:, j] = P[:, j] + learning_rate * (2 * eij * Q[i, :] - reg_lambda * P[:, j])
        P = P_new * (1/ number_of_restaurants)
        Q = Q_new * (1/ number_of_users)
        # calculate the training loss
        train_loss = 0
        for count in range(len(non_zero_idx[0])):
	        i = non_zero_idx[0][count]
	        j = non_zero_idx[1][count]
	        train_loss += pow(M[i,j] - np.dot(Q[i,:], P[:,j]), 2)
	        train_loss += reg_lambda * (np.sum(pow(Q[i, :], 2)) + np.sum(pow(P[:,j],2)))
	        
        # calculate the validation loss
        validation_loss = 0
        for count in range(len(val_values)):
	        i = val_idx[0][count]
	        j = val_idx[1][count]
	        validation_loss += pow(val_values[count] - np.dot(Q[i,:], P[:,j]), 2)
	        validation_loss += reg_lambda * (np.sum(pow(Q[i, :], 2)) + np.sum(pow(P[:, j],2)))
        if validation_loss < initial_val_loss:
            initial_val_loss = validation_loss
            best_Q = Q
            best_P = P
        if step % log_every == 0:
            print ('Iteration {}, training loss: {}, validation loss: {}'.format(step, train_loss, validation_loss))
        if step % eval_every == 0:
            if validation_losses:
                if abs(validation_loss - validation_losses[-1]) < 1:
                    temp_patience -= 1
                else:
                    temp_patience = patience
            train_losses.append(train_loss)
            validation_losses.append(validation_loss)
            if temp_patience == 0:
                converged_after = step - patience * eval_every
                break
            

    return best_Q, best_P, validation_losses, train_losses, converged_after

ratings = np.load("ratings.npy")
number_of_users = len(np.unique(ratings[:,0]))
number_of_restaurants = len(np.unique(ratings[:,1]))
M = sp.csr_matrix((ratings[:,2], (ratings[:,0], ratings[:,1])), 
                  shape=(number_of_users, number_of_restaurants))

M = cold_start_preprocessing(M, 10)
n_validation = 200
n_test = 200
# Split data
M_train, val_idx, test_idx, val_values, test_values = split_data(M, n_validation, n_test)
# Store away the nonzero indices of M before subtracting the row means.
nonzero_indices = M.nonzero()
# Remove user means.
M_shifted, user_means = shift_user_mean(M_train)

# Apply the same shift to the validation and test data.
val_values_shifted = val_values.copy()
test_values_shifted = test_values.copy()
for i in range(0, n_validation):
    val_values_shifted[i] = val_values_shifted[i] - user_means[val_idx[0][i]]
    test_values_shifted[i] = test_values_shifted[i] - user_means[test_idx[0][i]]

Q_g_sweep, P_g_sweep, val_l_g_sweep, tr_l_g_sweep, conv_g_sweep =  latent_factor_gradient_descent(M_shifted, nonzero_indices, 
                                                                                                   k=30, val_idx=val_idx,
                                                                                                   val_values=val_values_shifted, 
                                                                                                   reg_lambda=1, learning_rate=1e-1,
                                                                                                   init='svd', batch_size=-1,
                                                                                                   max_steps=10000, log_every=20, 
                                                                                                   eval_every=20)
