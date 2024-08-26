from utils import *
from scipy.linalg import sqrtm

import numpy as np
import matplotlib.pyplot as plt


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
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    # Randomly select a pair (user_id, question_id).
    i = \
        np.random.choice(len(train_data["question_id"]), 1)[0]

    c = train_data["is_correct"][i]
    n = train_data["user_id"][i]
    q = train_data["question_id"][i]
    u_n = u[n]
    z_q = z[q]

    # Compute the prediction error for the selected user-item pair.
    predicted_rating = np.dot(u_n, z_q)
    error = c - predicted_rating

    # Update the feature vectors based on the error.
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

    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    for i in range(0, num_iteration):
        update_u_z(train_data, lr, u, z)
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

    #####################################################################
    # TODO:                                                             #
    # (SVD) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    # To get the best accuray, we would do a loop.
    best_k = 0
    best_acc = 0
    res = []
    res1 = []
    for i in range(1, 27, 5):
        matrix_k = svd_reconstruct(train_matrix, i)
        acc = sparse_matrix_evaluate(val_data, matrix_k, 0.5)
        if acc > best_acc:
            best_acc = acc
            best_k = i
        res.append(i)
        res1.append(acc)
        # print(i, acc)
    plt.title("Validation Accuracy vs K-value(SVD)")
    plt.plot(res, res1, label="Validation Accuracy")
    plt.xlabel("k-value")
    plt.ylabel("Validation Accuracy(SVD)")
    plt.legend()
    plt.show()
    matrix_b = svd_reconstruct(train_matrix, best_k)
    print("Best K-value SVD:", best_k)
    # we have chosen the best accuray to be 6
    # Then we would calculate the test accuracy:
    print(f"Validation Accuracy SVD for k = {best_k}:", sparse_matrix_evaluate(val_data, matrix_b, 0.5))
    print(f"Test Accuracy for SVD k = {best_k}:", sparse_matrix_evaluate(test_data, matrix_b, 0.5))

    # part (b)
    # So one limitation of the SVD approach for this task is our method of handling missing entries
    # by inputting them with the average rating of the current item. This practice implicitly
    # assumes that all users are likely to rate the item close to its average rating, leading
    # to an over generalization of the item's appeal. Such an approach oversimplifies the diversity
    # of opinions on certain items. For example, an item that receives polarized reviews—half
    # being excellent and half being terrible would have the same mean rating as an item that
    # consistently receives average or normal reviews. By using the mean to fill in missing values, we
    # tend to smooth over such distinct distributions, masking the true variability in user
    # opinions and potentially distorting the item's profile in the analysis.

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # (ALS) Try out at least 5 different k and select the best k        #
    # using the validation set.                                         #
    #####################################################################
    b_k = 0
    b_acc = 0
    it = []
    res1 = []
    for i in range(1, 26, 5):
        als_mat = als(train_data, i, 0.06, 80000)
        acc = sparse_matrix_evaluate(val_data, als_mat, 0.5)
        if acc > b_acc:
            b_acc = acc
            b_k = i
        # print(i, acc)
        it.append(i)
        res1.append(acc)
    plt.plot(it, res1, linestyle="dashed")
    plt.title("Validation Accuracy vs K-value(SGD)")
    plt.xlabel("k-value")
    plt.ylabel("Validation Accuracy(SGD)")
    plt.show()
    print("Best K for ALS: ", 6)
    als_mat = als(train_data, 6, 0.06, 80000)
    # we have chosen the best accuray to be 6
    # Then we would calculate the test accuracy:
    print(f"Validation Accuracy ALS for k = {6}:", sparse_matrix_evaluate(val_data, als_mat, 0.5))
    print(f"Test Accuracy ALS for k = {6}:", sparse_matrix_evaluate(test_data, als_mat, 0.5))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    u = np.random.uniform(low=0, high=1 / np.sqrt(6),
                          size=(len(set(train_data["user_id"])), 6))
    z = np.random.uniform(low=0, high=1 / np.sqrt(6),
                          size=(len(set(train_data["question_id"])), 6))
    line_t = []
    line_v = []
    iter = []
    for iteration in range(1, 80000, 1000):
        for _ in range(0, iteration):
            u, z = update_u_z(train_data, 0.06, u, z)
        line_t.append(squared_error_loss(train_data, u, z))
        line_v.append(squared_error_loss(val_data, u, z))
        iter.append(iteration)
    # print(line_t)
    # print(line_v)
    plt.plot(iter, line_t, linestyle="dashed", label="Training accuracy")
    plt.plot(iter, line_v, linestyle="solid", label="Validation accuracy")
    plt.title("Squared Loss vs Iteration (SGD)")
    plt.xlabel("Iteration")
    plt.ylabel("Squared Loss (SGD)")
    plt.legend()
    plt.ylim(0, 10000)
    plt.show()

if __name__ == "__main__":
    main()