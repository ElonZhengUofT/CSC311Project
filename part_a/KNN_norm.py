from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from utils import *


def knn_impute_by_user(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    student similarity. Return the accuracy on valid_data.

    See https://scikit-learn.org/stable/modules/generated/sklearn.
    impute.KNNImputer.html for details.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    return acc


def knn_impute_by_item(matrix, valid_data, k):
    """ Fill in the missing values using k-Nearest Neighbors based on
    question similarity. Return the accuracy on valid_data.

    :param matrix: 2D sparse matrix
    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param k: int
    :return: float
    """
    #####################################################################
    matrix_transposed = matrix.T
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat_transposed = nbrs.fit_transform(matrix_transposed)
    # Transpose back the imputed matrix to original orientation
    mat = mat_transposed.T
    acc = sparse_matrix_evaluate(valid_data, mat)
    print("Validation Accuracy: {}".format(acc))
    # Implement the function as described in the docstring.             #
    #####################################################################
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return acc


def main():
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    print("Sparse matrix:")
    print(sparse_matrix)
    print("Shape of sparse matrix:")
    print(sparse_matrix.shape)
    k_values = [1, 6, 11, 16, 21, 26]

    # Run user-based collaborative filtering for each k
    accuracies = []
    accuracies1 = []
    for k in k_values:
        accuracy = knn_impute_by_user(sparse_matrix, val_data, k)
        accuracy1 = knn_impute_by_item(sparse_matrix, val_data, k)
        accuracies1.append(accuracy1)
        accuracies.append(accuracy)

    # Plot the validation accuracy as a function of k
    plt.plot(k_values, accuracies, marker='o')
    plt.title('Validation Accuracy vs. k for User-Based Collaborative Filtering')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.show()
    plt.plot(k_values, accuracies1, marker='o')
    plt.title('Validation Accuracy vs. k for Item-Based Collaborative Filtering')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.grid(True)
    plt.show()
    #####################################################################
    # Compute the validation accuracy for each k. Then pick k* with     #
    # the best performance and report the test accuracy with the        #
    # chosen k*.                                                        #
    #####################################################################
    knn_impute_by_item(sparse_matrix, test_data, 30)  #
    print(sparse_matrix)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
