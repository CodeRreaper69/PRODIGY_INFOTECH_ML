# Hand Gesture Recognition using SVM

This project implements a hand gesture recognition system using a Support Vector Machine (SVM) model. The model is trained on a dataset containing hand gesture images, and it achieves a 100% accuracy score.

## Dataset

The dataset used for training and testing the model can be found on Kaggle: [Hand Gesture Recognition Dataset](https://www.kaggle.com/datasets/aryarishabh/hand-gesture-recognition-dataset)

## Project Structure

- `gesture.py`: Script for collecting and storing grayscale hand gesture images in a pickled format (`data.pickle`).
- `train.py`: Script for training an SVM model on the collected data and testing its accuracy.
- `train_test.py`: Real-time hand gesture recognition using the trained SVM model on live video feed.

## Instructions

1. **Collect Data**: Run `gesture.py` to capture and store grayscale hand gesture images in the `data.pickle` file.

    ```bash
    python gesture.py
    or
     python3 gesture.py
    ```

2. **Train the Model**: Execute `train.py` to train the SVM model on the collected data.

    ```bash
    python train.py
    or
    python3 train.py
    ```

3. **Test the Model in Real-time**: Run `train_test.py` for real-time hand gesture recognition using the trained model on live video feed.

    ```bash
    python train_test.py
     python3 train_test.py
    ```

## Dependencies

- OpenCV
- NumPy
- Matplotlib
- scikit-learn

Install dependencies using:

```bash
pip install opencv-python numpy matplotlib scikit-learn

