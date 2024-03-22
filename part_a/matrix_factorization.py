from utils import *
from scipy.linalg import sqrtm

import numpy as np


def svd_reconstruct(matrix, k):
    """ Given the matrix, perform singular value decomposition
    to reconstruct the matrix.

    :param matrix: 2D sparse matrix
    :param k: int
    :return: 2D matrix
    """
    # First, you need to fill in the missing values (NaN) to perform SVD.
    # Fill in the missing values using the average on the current item.
    # Note that there are many options to do fill in the
    # missing values (e.g. fill with 0).
    new_matrix = matrix.copy()
    mask = np.isnan(new_matrix)
    masked_matrix = np.ma.masked_array(new_matrix, mask)
    item_means = np.mean(masked_matrix, axis=0)
    new_matrix = masked_matrix.filled(item_means)

    # Next, compute the average and subtract it.
    item_means = np.mean(new_matrix, axis=0)
    mu = np.tile(item_means, (new_matrix.shape[0], 1))
    new_matrix = new_matrix - mu

    # Perform SVD.
    Q, s, Ut = np.linalg.svd(new_matrix, full_matrices=False)
    s = np.diag(s)

    # Choose top k eigenvalues.
    s = s[0:k, 0:k]
    Q = Q[:, 0:k]
    Ut = Ut[0:k, :]
    s_root = sqrtm(s)

    # Reconstruct the matrix.
    reconst_matrix = np.dot(np.dot(Q, s_root), np.dot(s_root, Ut))
    reconst_matrix = reconst_matrix + mu
    return np.array(reconst_matrix)


def squared_error_loss(data, u, z):
    """ Return the squared-error-loss given the data.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param u: 2D matrix
    :param z: 2D matrix
    :return: float
    """
    loss = 0
    for i, q in enumerate(data["question_id"]):
        loss += (data["is_correct"][i]
                 - np.sum(u[data["user_id"][i]] * z[q])) ** 2.
    return 0.5 * loss


def update_u_z(train_data, lr, u, z):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    """
    #####################################################################                                                           #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]

    # Compute the gradient of the loss function with respect to u_n and z_q.
    grad_u = (c - np.dot(u[n], z[q])) * z[q]
    grad_z = (c - np.dot(u[n], z[q])) * u[n]

    # Update u_n and z_q.
    u[n] = u[n] + lr * grad_u
    z[q] = z[q] + lr * grad_z


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration):
    """ Performs ALS algorithm, here we use the iterative solution - SGD 
    rather than the direct solution.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :param lr: float
    :param num_iteration: int
    :return: 2D reconstructed Matrix.
    """
    # Initialize u and z
    u = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["user_id"])), k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(k),
                          size=(len(set(train_data["question_id"])), k))

    #####################################################################                                                            #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Perform ALS.
    for i in range(num_iteration):
        # Update U
        for n in range(u.shape[0]):
            relevant_samples = [i for i in range(len(train_data["user_id"])) if
                                train_data["user_id"][i] == n]
            grad_u = np.zeros(k)
            for i in relevant_samples:
                c = train_data["is_correct"][i]
                q = train_data["question_id"][i]
                prediction = np.dot(u[n], z[q])
                grad_u += (c - prediction) * z[q]
            u[n] += lr * grad_u

        # Update Z
        for q in range(z.shape[0]):
            relevant_samples = [i for i in range(len(train_data["question_id"]))
                                if train_data["question_id"][i] == q]
            grad_z = np.zeros(k)
            for i in relevant_samples:
                c = train_data["is_correct"][i]
                n = train_data["user_id"][i]
                prediction = np.dot(u[n], z[q])
                grad_z += (c - prediction) * u[n]
            z[q] += lr * grad_z

    mat = np.dot(u, z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat


def main():
    train_matrix = load_train_sparse("../data").toarray()
    train_data = load_train_csv("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    best_k_svd = 0
    best_acc_svd = 0
    for k in [1, 2, 5, 10, 20]:
        reconst_matrix = svd_reconstruct(train_matrix, k)
        # Evaluate the accuracy of the reconstructed matrix.
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        print("Validation Accuracy: {} with k = {}".format(acc, k))
        if acc > best_acc_svd:
            best_acc_svd = acc
            best_k_svd = k
    print("Best k: {} with accuracy: {}".format(best_k_svd, best_acc_svd))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################                                                          #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    best_k_als = 0
    best_acc_als = 0
    for k in [1, 2, 5, 10, 20]:
        reconst_matrix = als(train_data, k, 0.01, 10)
        # Evaluate the accuracy of the reconstructed matrix.
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        print("Validation Accuracy: {} with k = {}".format(acc, k))
        if acc > best_acc_als:
            best_acc_als = acc
            best_k_als = k
    print("Best k: {} with accuracy: {}".format(best_k_als, best_acc_als))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
