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
    for k in [1,2,5,10,20]:
        for lr in [0.01, 0.1, 0.5, 1]:
            for num_iteration in range(1, 80000, 2000):
                reconst_matrix, u, z = als(train_data, k, 0.01, 100)
                # Evaluate the accuracy of the reconstructed matrix.
                acc = sparse_matrix_evaluate(val_data, reconst_matrix)
                if acc > best_acc_als:
                    best_acc_als = acc
                    best_k_als = k
                    best_lr_als = lr
                    best_iteration_als = num_iteration
                    best_model = reconst_matrix
        print("k: {}".format(k))
    print("Best k: {},Best lr: {}, Best iteration: {} with accuracy: {}".format(best_k_als, best_lr_als, best_iteration_als, best_acc_als))
    t_acc = sparse_matrix_evaluate(test_data, best_model)
    print("Test Accuracy: {}".format(t_acc))

    best_k_als = 0
    best_acc_als = 0
    for k in range(1, 100):
        reconst_matrix, u, z = als(train_data, k, best_lr_als, best_iteration_als)
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        if acc > best_acc_als:
            best_acc_als = acc
            best_k_als = k
            best_model = reconst_matrix
    print("Best k: {}, with accuracy: {}".format(best_k_als, best_acc_als))
    t_acc = sparse_matrix_evaluate(test_data, best_model)
    print("Test Accuracy: {}".format(t_acc))

    best_lr_als = 0
    best_acc_als = 0
    for lr in np.arange(0.01, 1, 0.01):
        reconst_matrix, u, z = als(train_data, best_k_als, lr, best_iteration_als)
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        if acc > best_acc_als:
            best_acc_als = acc
            best_lr_als = lr
            best_model = reconst_matrix

    print("Best lr: {}, with accuracy: {}".format(best_lr_als, best_acc_als))
    t_acc = sparse_matrix_evaluate(test_data, best_model)
    print("Test Accuracy: {}".format(t_acc))

    best_iteration_als = 0
    best_acc_als = 0
    for num_iteration in range(1, 1000, 20):
        reconst_matrix, u, z = als(train_data, best_k_als, best_lr_als, num_iteration)
        acc = sparse_matrix_evaluate(val_data, reconst_matrix)
        if acc > best_acc_als:
            best_acc_als = acc
            best_iteration_als = num_iteration
            best_model = reconst_matrix
    print("Best iteration: {}, with accuracy: {}".format(best_iteration_als, best_acc_als))
    t_acc = sparse_matrix_evaluate(test_data, best_model)
    print("Final Model Validation Accuracy: {}, Test Accuracy: {}".format(best_acc_als, t_acc))


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