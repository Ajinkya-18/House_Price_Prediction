# 🏠 California Housing Price Prediction

This project aims to build a **regression model** to predict **median house prices** using the **California Housing Prices** dataset. The project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation using multiple regression models.

---

## 📂 Dataset

- **Source**: [Kaggle - California Housing Prices] (https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Description**: The dataset contains information collected from the 1990 California census. It includes features such as population, number of households, median income, housing median age, and more.

---

## 📌 Project Structure

├── notebooks/ HousePricePrediction.ipynb # Main notebook with all steps
├── README.md # Project documentation
├── data/ housing.csv # Dataset (downloaded from Kaggle)
│── models/ scalers and trained models (*.pkl)s

---

## 🧠 Project Highlights

### 1. 🧹 Data Preprocessing
- Feature encoding (`ocean_proximity`)

### 2. 📊 Exploratory Data Analysis (EDA)
- Correlation analysis and visualizations
- Scatter plots for model performance evaluation
- Histograms and boxplots

### 3. 🛠️ Feature Engineering
- Added new features: `rooms_per_household`, `bedrooms_per_room`, `population_per_household`
- Feature scaling using `QuantileTransformer` and RandomSearchCV pipelines

### 4. 🤖 Model Training & Evaluation
Trained and evaluated multiple models:
- **Linear Regression**
- **KNeighbors Regressor**
- **Random Forest Regressor**
- **HistGradientBoosting Regressor**
- **Grid Search** for hyperparameter tuning

Used metrics like:
- R2_score
---

## ⚙️ Setup Instructions

### 1. Clone the Repository
git clone https://github.com/your-username/California-Housing-Prediction.git
cd California-Housing-Prediction

### 2. Download Dataset
Download the dataset from Kaggle and place housing.csv in the data/ folder.

### 3. Create and Activate a Conda Environment
conda create -n house-price-env python=3.10 -y
conda activate house-price-env

### 4. Install Required Dependencies
You can install packages using the following command:
pip install -r requirements.txt

Or if you have an environment.yml file:
conda env create -f environment.yml
conda activate house-price-env

5. Run the Notebook
Use Jupyter to open and run the notebook:
jupyter notebook HousePricePrediction.ipynb


## 📈 Results
The HistGradientBoosting Regressor performed the best among all models with the highest R2_score. Feature engineering, Feature Selection and pipeline integration significantly improved model performance.

### 🙏 Acknowledgements
The dataset is provided by Kaggle.

#### 📌 Author
Ajinkya Tamhankar
