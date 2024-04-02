from utils import *
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

import numpy as np
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


def update_u_z(train_data, lr, u, z, Lambda):
    """ Return the updated U and Z after applying
    stochastic gradient descent for matrix completion.

    :param train_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param u: 2D matrix
    :param z: 2D matrix
    :return: (u, z)
    The loss function is given by:
    L = 0.5 * (c - u_n^T * z_q)^2
    where c is the correctness of the answer, u_n is the user vector, and z_q is the question vector.
    Now with L2 regularization, the loss function becomes:
    L = 0.5 * (c - u_n^T * z_q)^2 + 0.5 * lambda * (||u_n||^2 + ||z_q||^2)
    and the gradient of the loss function with respect to u_n and z_q is given by:
    grad_u = (c - u_n^T * z_q) * z_q - lambda * u_n
    grad_z = (c - u_n^T * z_q) * u_n - lambda * z_q
    where lambda is the regularization parameter.
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
    grad_u = (c - np.dot(u[n], z[q])) * z[q] - Lambda * u[n]
    grad_z = (c - np.dot(u[n], z[q])) * u[n] - Lambda * z[q]

    # Update u_n and z_q.
    u[n] = u[n] + lr * grad_u
    z[q] = z[q] + lr * grad_z


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return u, z


def als(train_data, k, lr, num_iteration, Lambda):
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
        u, z = update_u_z(train_data, lr, u, z, Lambda)

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
    #####################################################################                                                          #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################

    best_k_als = 0
    best_lr_als = 0
    best_iteration_als = 0
    best_Lambda = 0

    best_acc_als = 0
    for k in [1, 2, 5, 10, 20]:
        for lr in [0.01, 0.1, 0.5, 1]:
            for num_iteration in range(1, 1000, 20):
                for Lambda in [0.01, 0.1, 0.5, 1]:
                    reconst_matrix, u, z = als(train_data, k, 0.01, 100, Lambda)
                    # Evaluate the accuracy of the reconstructed matrix.
                    acc = sparse_matrix_evaluate(val_data, reconst_matrix)
                    if acc > best_acc_als:
                        best_acc_als = acc
                        best_k_als = k
                        best_lr_als = lr
                        best_iteration_als = num_iteration
                        best_Lambda = Lambda
        k_reconst_matrix = als(train_data, k, best_lr_als, best_iteration_als, best_Lambda)
        print("Validation Accuracy: {} with k = {}".format(acc, k))
    print("Best k: {},Best lr: {}, Best iteration: {} with accuracy: {}, Best Lambda: {}".format(best_k_als, best_lr_als, best_iteration_als, best_acc_als, Lambda))
    final_reconst_matrix, u, z = als(train_data, best_k_als, best_lr_als, best_iteration_als, best_Lambda)
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
        plt.show()

    num_iteration = [1, 5, 10, 20, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    train_losses = []
    val_losses = []
    best_model = None
    best_val_acc = 0

    for i in num_iteration:
        print("Iteration: {}".format(i))
        reconst_matrix , u, z = als(train_data, best_k_als, best_lr_als, i, best_Lambda)

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

