# house-price-ML
House Price Prediction with XGBoost
This project focuses on predicting house prices using the XGBoost algorithm, leveraging Python libraries like Pandas, Matplotlib, Seaborn, and Scikit-learn.

Overview
The project follows these major steps:

Data Loading and Inspection:

Loads the dataset (house_tiny.csv).
Prints the first few rows, shape, info, and statistical summary of the dataset.
Checks for missing values and handles them appropriately.
Data Preprocessing:

Fills missing values in 'Item_Weight' with the mean.
Fills missing values in 'Outlet_Size' based on the mode grouped by 'Outlet_Type'.
Standardizes values in 'Item_Fat_Content' for consistency.
Encodes categorical features using LabelEncoder.
Data Visualization:

Visualizes distributions and counts of various features using histograms and count plots.
Provides insights into 'Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Outlet_Sales', 'Outlet_Establishment_Year', 'Item_Fat_Content', 'Item_Type', and 'Outlet_Size'.
Model Building and Evaluation:

Prepares data for modeling by splitting into training and testing sets.
Utilizes XGBoostRegressor to train a model on the prepared data.
Evaluates model performance using R-squared score on both training and test datasets.
Technologies Used
Python Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, XGBoost.
Development Environment: Jupyter Notebook, Python scripts.
File Structure
house_tiny.csv: Dataset containing house-related features.
README.md: Project documentation providing an overview of the project, steps involved, and technologies used.
visualization.py: Python script for data visualization.
modeling.py: Python script for data preprocessing, modeling, and evaluation.
Setup Instructions
Clone the repository:
git clone https://github.com/yourusername/house-price-prediction.git
cd house-price-prediction

Install dependencies:
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
Run the scripts or notebooks to reproduce the results or modify as per your requirements.

Conclusion
This project serves as a comprehensive example of data preprocessing, visualization, modeling, and evaluation using XGBoost for predicting house prices based on various features.
