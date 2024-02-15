# House Price Prediction

This project involves predicting house prices based on features such as the number of bedrooms, bathrooms, and square footage. The dataset used for this project is named "house_price.csv".

## Dataset Overview

- The dataset contains information about houses, including features like bedrooms, bathrooms, square footage, etc.
- It's recommended to run the `data_insights` function to get an overview of the dataset.

## Data Visualization

- The `data_vis` function generates count plots to visualize relationships between different features in the dataset.

## Model Training and Prediction

1. **Data Preprocessing:**
   - The dataset is split into features (`X`) and the target variable (`y`).
   - Missing values are handled if necessary.
   - Additional features like 'total_sqft' are created.

2. **Model Training:**
   - The dataset is split into training and testing sets.
   - The target variable is normalized using the mean and standard deviation of the training set.
   - A linear regression model is trained using the normalized target variable.

3. **Model Evaluation:**
   - The model is evaluated using Mean Squared Error (MSE) and R-squared value.
   - High MSE indicates prediction errors, and a higher R-squared value suggests better model fit.

4. **User Input and Prediction:**
   - Users can input features (square footage, bedrooms, bathrooms) to get a predicted house price.
   - The user input is normalized using training set statistics before making predictions.

## How to Run

1. Ensure you have the required Python libraries installed (`pandas`, `matplotlib`, `seaborn`, `scikit-learn`).
   ```bash
   pip install -r requirements.txt
   ```

2. Clone the repository:
   ```bash
   git clone https://github.com/CodeRreaper69/PRODIGY_INFOTECH_ML.git
   ```

3. Navigate to the project directory:
   ```bash
   cd PRODIGY_INFOTECH_ML/PRODIGY_ML_01
   ```

4. Run the script:
   ```bash
   python house_price_predict.py
   or
   python3 house_price_predict.py
   ```
5. The link to the dataset is here - https://www.kaggle.com/datasets/shree1992/housedata

