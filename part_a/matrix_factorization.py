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
    predicted_rating = np.dot(u[n], z[q])
    error = c - predicted_rating

    # Update u_n and z_q.
    u[n] += lr * error * z[q]
    z[q] += lr * error * u[n]

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
    ini_lr = 0.05
    ini_num_iteration = 80000
    ini_k = 11
    b_k = 0
    b_lr = 0
    b_num_iteration = 0
    b_acc = 0
    it = []
    res1 = []

    reconst_matrix = als(train_data, ini_k, ini_lr, ini_num_iteration)
    b_acc = sparse_matrix_evaluate(val_data, reconst_matrix)
    print("Validation Accuracy ALS for k = {}:".format(ini_k), b_acc)

    for i in range(1, 26, 5):
        reconst_matrix = als(train_data, i, ini_lr, ini_num_iteration)
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        if acc > b_acc:
            b_acc = acc
            b_k = i
        it.append(i)
        res1.append(acc)
        print("k: {}, acc: {}".format(i, acc))
    if b_k == 0:
        b_k = ini_k
    plt.plot(it, res1)
    plt.title("Validation Accuracy vs K")
    plt.xlabel("k-value")
    plt.ylabel("Validation Accuracy")
    plt.savefig('Validation Accuracy vs K.jpg')
    plt.show()
    print("Best K-value SGD:", b_k)
    als_mat = als(train_data, b_k, ini_lr, ini_num_iteration)
    print("Validation Accuracy ALS for k = {}:".format(b_k), sparse_matrix_evaluate(val_data, als_mat, 0.5))
    print("Test Accuracy ALS for k = {}:".format(b_k), sparse_matrix_evaluate(test_data, als_mat, 0.5))

    it = []
    v_a = []
    for lr in np.arange(0.01,0.26, 0.05):
        reconst_matrix = als(train_data, b_k, lr, ini_num_iteration)
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        if acc > b_acc:
            b_acc = acc
            b_lr = round(float(lr), 2)
        print("lr: {}".format(lr))
        it.append(lr)
        v_a.append(acc)
    if b_lr == 0:
        b_lr = ini_lr

    plt.plot(it, v_a)
    plt.title("Validation Accuracy vs Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Validation Accuracy")
    plt.savefig('Validation Accuracy vs Learning Rate.jpg')
    plt.show()


    it = []
    v_a = []
    for num_iteration in range(1, 100001,10000):
        reconst_matrix = als(train_data, b_k, b_lr, num_iteration)
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        if acc > b_acc:
            b_acc = acc
            b_num_iteration = num_iteration
        print("num_iteration: {}".format(num_iteration))
        it.append(num_iteration)
        v_a.append(acc)
    if b_num_iteration == 0:
        b_num_iteration = ini_num_iteration

    plt.plot(it, v_a)
    plt.title("Validation Accuracy vs Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Validation Accuracy")
    plt.savefig('Validation Accuracy vs Number of Iterations.jpg')
    plt.show()

    als_mat = als(train_data, b_k, b_lr, b_num_iteration)
    print("Validation Accuracy ALS for k = {}, lr = {}, num_iteration = {}:".format(b_k, b_lr, b_num_iteration),
          sparse_matrix_evaluate(val_data, als_mat, 0.5))
    print("Test Accuracy ALS for k = {}, lr = {}, num_iteration = {}:".format(b_k, b_lr, b_num_iteration),
          sparse_matrix_evaluate(test_data, als_mat, 0.5))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    u = np.random.uniform(low=0, high=1 / np.sqrt(b_k),
                          size=(len(set(train_data["user_id"])), b_k))
    z = np.random.uniform(low=0, high=1 / np.sqrt(b_k),
                          size=(len(set(train_data["question_id"])), b_k))
    line_t = []
    line_v = []
    iter = []
    for iteration in range(1, 80000, 1000):
        for _ in range(0, iteration):
            u, z = update_u_z(train_data, b_lr, u, z)
        line_t.append(squared_error_loss(train_data, u, z))
        line_v.append(squared_error_loss(val_data, u, z))
        iter.append(iteration)
    # print(line_t)
    # print(line_v)
    plt.plot(iter, line_t, linestyle="solid", label="Training loss")
    plt.title("Training Loss vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Squared Loss")
    plt.legend()
    plt.ylim(0, 13000)
    plt.savefig('losses vs iteration.jpg')
    plt.show()

    plt.plot(iter, line_v, linestyle="dashed", label="Validation loss")
    plt.title("Validation Loss vs Iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Squared Loss")
    plt.legend()
    plt.ylim(0, 2000)
    plt.savefig('validation loss vs iteration.jpg')
    plt.show()


    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
