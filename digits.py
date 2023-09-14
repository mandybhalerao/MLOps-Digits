# This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# Import datasets, classifiers and performance metrics
from utils import preprocess_data, tune_hparams, split_train_dev_test,read_digits,predict_and_eval


# The digits dataset consists of 8x8 pixel images of digits. The images attribute of the dataset stores 8x8 arrays of grayscale values for each image. We will use these arrays to visualize the first 4 images. The target attribute of the dataset stores the digit each image represents and this is included in the title of the 4 plots below.
# Note: if we were working from image files (e.g., ‘png’ files), we would load them using matplotlib.pyplot.imread.

# 1. Data Loading

x,y = read_digits()

test_sizes = [0.1, 0.2, 0.3]
dev_sizes = [0.1, 0.2, 0.3]

for test_size in test_sizes:
    for dev_size in dev_sizes:
        # 3. Data splitting
        X_train, X_test,X_dev, y_train, y_test,y_dev = split_train_dev_test(x, y, test_size=test_size, dev_size=dev_size);

        # 4. Data Preprocessing
        X_train = preprocess_data(X_train)
        X_test = preprocess_data(X_test)
        X_dev = preprocess_data(X_dev)

        gama_ranges = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        C_ranges = [0.1,1,2,5,10]
        list_of_all_param_combination = [{'gamma': gamma, 'C': C} for gamma in gama_ranges for C in C_ranges]

        # Predict the value of the digit on the test subset
        # 6.Predict and Evaluate 
        best_hparams, best_model, best_accuracy = tune_hparams(X_train, y_train, X_dev, y_dev, list_of_all_param_combination)
        accuracy_test = predict_and_eval(best_model, X_test, y_test)
        accuracy_dev = predict_and_eval(best_model, X_dev, y_dev)
        accuracy_train = predict_and_eval(best_model, X_train, y_train)
        print(f"test_size={test_size} dev_size={dev_size} train_size={1- (dev_size+test_size)} train_acc={accuracy_train} dev_acc={accuracy_dev} test_acc={accuracy_test}")
        print(f"best_gamma={best_hparams['gamma']},best_C={best_hparams['C']}")







