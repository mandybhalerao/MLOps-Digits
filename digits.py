# This example shows how scikit-learn can be used to recognize images of hand-written digits, from 0-9.

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause


# Import datasets, classifiers and performance metrics
from utils import preprocess_data, train_model, split_data,read_digits,predict_and_eval

# The digits dataset consists of 8x8 pixel images of digits. The images attribute of the dataset stores 8x8 arrays of grayscale values for each image. We will use these arrays to visualize the first 4 images. The target attribute of the dataset stores the digit each image represents and this is included in the title of the 4 plots below.
# Note: if we were working from image files (e.g., ‘png’ files), we would load them using matplotlib.pyplot.imread.

# 1. Data Loading


x,y = read_digits()

# 3. Data splitting
X_train, X_test, y_train, y_test = split_data(x, y, test_size=0.3);

# 4. Data Preprocessing
X_train = preprocess_data(X_train)
X_test = preprocess_data(X_test)

# 5. Train the data
model = train_model(X_train, y_train, {'gamma': 0.001}, model_type='svm')



# Predict the value of the digit on the test subset
# 6.Predict and Evaluate 
predicted = predict_and_eval(model, X_test, y_test)





