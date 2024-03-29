**Mall Customer Segmentation**

**Introduction:**
This project focuses on segmenting mall customers based on their annual income and spending score using the K-Means clustering algorithm. The dataset used for this analysis is named "Mall_Customers.csv," which contains information about customers' annual income, spending score, and customer ID.

**Files:**
1. **mall_customer_segmentation.py:**
   - This Python script contains the code for data analysis, visualization, and clustering using the K-Means algorithm.
   - It reads the "Mall_Customers.csv" dataset using the Pandas library.
   - Provides insights into the dataset, including basic statistics, shape, column names, and null values.
   - Performs data visualization using seaborn, displaying count plots for various combinations of fields.
   - Standardizes relevant features ('Annual Income (k$)', 'Spending Score (1-100)') and determines the optimal number of clusters using the elbow method.
   - Applies K-Means clustering with the optimal number of clusters and assigns each customer to a specific cluster.
   - Visualizes the clustering results using a bar plot.
2. **Mall_Customer.csv**
   - Dataset for this prediction
   - dataset link - https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python
3. **requirements.txt**
   - required librarires

**Instructions:**
1. Clone the repository from GitHub:
   ```bash
   git clone https://github.com/CodeRreaper69/PRODIGY_INFOTECH_ML.git
   ```

2. Navigate to the "PRODIGY_INFOTECH_ML/PRODIGY_ML_2" directory:
   ```bash
   cd PRODIGY_INFOTECH_ML/PRODIGY_ML_2
   ```

3. Run the Python script "mall_customer_segmentation.py" using a Python interpreter:
   ```bash
   python mall_customer_segmentation.py
   ```

4. Observe the insights, visualizations, and clustering results generated by the script.

**Note:**
- Ensure you have the required Python libraries installed, including Pandas, NumPy, Matplotlib, Seaborn, and scikit-learn.
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   or
   pip install -r requirements.txt
   ```

- The script reads the "Mall_Customers.csv" dataset, analyzes customer behavior, and segments them into clusters based on income and spending score.
