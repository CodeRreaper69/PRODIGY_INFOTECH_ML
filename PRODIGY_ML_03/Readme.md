**Dog and Cat Classification using SVM**

**Introduction:**
This project aims to classify images of dogs and cats using a Support Vector Machine (SVM) algorithm. The images are preprocessed and converted into a portable format using NumPy arrays. The SVM model is trained on these arrays to make predictions on new images.

**Files:**
1. **image_svm.py:**
   - Responsible for preprocessing the images and creating a dataset in the form of a pickled NumPy array ('data.pickle').
   - Images are resized to 50x50 pixels and converted to grayscale.
   - Each image is flattened into a 1D array, and corresponding labels (0 for dogs, 1 for cats) are assigned.
   - The dataset is saved using pickle for portability.

2. **svm_class.py:**
   - Loads the preprocessed data from 'data.pickle'.
   - Splits the data into training and testing sets using `train_test_split` from scikit-learn.
   - Trains an SVM model with a polynomial kernel on the training set.
   - Calculates and prints the accuracy of the model on the test set.
   - Allows the user to input the path of an image for prediction.
   - Displays the image and predicts whether it is a dog or a cat using the trained SVM model.

**Instructions:**
1. Ensure you have the required libraries installed, including NumPy, OpenCV, Matplotlib, and scikit-learn.
   ```bash
   pip install numpy opencv-python matplotlib scikit-learn
   or
   pip install -r requirements.txt
   ```

2. Run `image_svm.py` to preprocess the images and create the pickled dataset.
   ```bash
   python image_svm.py
   or python3 image_svm.py
   ```

3. Run `svm_class.py` to train the SVM model, evaluate accuracy, and make predictions on user-provided images.
   ```bash
   python svm_class.py
   or
   python3 svm_class.py
   ```

4. Enter the path of an image when prompted and observe the model's prediction.
5. Dataset link - https://www.kaggle.com/datasets/tongpython/cat-and-dog

