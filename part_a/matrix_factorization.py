from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

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
    i = np.random.choice(len(train_data["question_id"]), 1)[0]

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
    for i in range(num_iteration):
        u, z = update_u_z(train_data, lr, u, z)

    mat = np.dot(u, z.T)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return mat, u, z


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
    final_reconst_matrix = svd_reconstruct(train_matrix, best_k_svd)
    t_acc = sparse_matrix_evaluate(test_data, final_reconst_matrix)
    print("Test Accuracy: {}".format(t_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    #####################################################################                                                          #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    best_k_als = 0
    best_lr_als = 0
    best_iteration_als = 0

    best_acc_als = 0
    for k in [1, 2, 5, 10, 20]:
        for lr in [0.01, 0.1, 0.5, 1]:
            for num_iteration in range(1, 1000, 20):
                reconst_matrix, u, z = als(train_data, k, 0.01, 100)
                # Evaluate the accuracy of the reconstructed matrix.
                acc = sparse_matrix_evaluate(val_data, reconst_matrix)
                if acc > best_acc_als:
                    best_acc_als = acc
                    best_k_als = k
                    best_lr_als = lr
                    best_iteration_als = num_iteration
        k_reconst_matrix = als(train_data, k, best_lr_als, best_iteration_als)
        print("Validation Accuracy: {} with k = {}".format(acc, k))
    print("Best k: {},Best lr: {}, Best iteration: {} with accuracy: {}".format(best_k_als, best_lr_als, best_iteration_als, best_acc_als))
    final_reconst_matrix, u, z = als(train_data, best_k_als, best_lr_als, best_iteration_als)
    t_acc = sparse_matrix_evaluate(test_data, final_reconst_matrix)
    print("Test Accuracy: {}".format(t_acc))
    #####################################################################
    # With your chosen hyperparameters, plot the training and validation squared-error
    # losses as a function of iteration. Also, report the validation accuracy and test accuracies
    # for your final model.
    def plot_losses(train_losses, val_losses, num_iteration):
        """Plot training and validation losses."""
        plt.plot(num_iteration,train_losses,label='Training Loss')
        plt.plot(num_iteration,val_losses, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Squared Error Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.savefig('losses vs iteration.png')
        plt.show()

    num_iteration = [1, 5, 10, 20, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    train_losses = []
    val_losses = []
    best_model = None
    best_val_acc = 0

    for i in num_iteration:
        print("Iteration: {}".format(i))
        reconst_matrix , u, z = als(train_data, best_k_als, best_lr_als, i)

        train_loss = squared_error_loss(train_data, u, z)
        val_loss = squared_error_loss(val_data, u, z)


        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = reconst_matrix
    plot_losses(train_losses, val_losses, num_iteration)
    test_acc = sparse_matrix_evaluate(test_data, best_model)
    print("Validation Accuracy: {}".format(test_acc))


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
