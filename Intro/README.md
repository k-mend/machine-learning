# 🚗 Car Fuel Efficiency Data Analysis  

This repository contains a Jupyter Notebook (`module 1.ipynb`) that explores **data wrangling and statistical analysis** using **Pandas** and **NumPy**. The notebook demonstrates how to clean, inspect, and analyze tabular data from the dataset `car_fuel_efficiency.csv`.  

## 📚 Contents  

The notebook covers the following topics:  

- **Library imports & setup**  
  - Using `pandas` and `numpy` for data analysis  
- **Loading data**  
  - Reading `car_fuel_efficiency.csv` with `pd.read_csv()`  
- **Data exploration**  
  - Viewing dataset shape and first rows with `.shape` and `.head()`  
  - Inspecting unique values with `.nunique()`  
  - Handling missing data using `.isnull().sum()` and `.fillna()`  
- **Filtering & subsetting**  
  - Selecting cars by region (e.g., `origin == 'Asia'`)  
  - Finding maximum fuel efficiency per region  
- **Statistical operations**  
  - Calculating median and mode values for columns  
  - Replacing missing values with median/mode  
- **Basic NumPy operations**  
  - Demonstrating arrays, element-wise operations, and linear algebra  

## 🛠️ Requirements  

To run the notebook, install the following dependencies:  

```bash
conda create -n datatools python=3.10 -y
conda activate datatools
conda install numpy pandas jupyterlab -y
```

Alternatively, with `pip`:  

```bash
pip install numpy pandas jupyterlab
```

## ▶️ Usage  

1. Clone the repository:  

   ```bash
   git clone https://github.com/your-username/car-fuel-efficiency.git
   cd car-fuel-efficiency
   ```

2. Launch Jupyter Lab:  

   ```bash
   jupyter lab
   ```

3. Open **`module 1.ipynb`** and run the cells interactively.  

## 📊 Dataset  

The dataset `car_fuel_efficiency.csv` includes attributes such as:  

- `fuel_type` – type of fuel used by the car  
- `origin` – country/region of manufacture  
- `horsepower` – engine horsepower (with missing values handled)  
- `fuel_efficiency_mpg` – miles per gallon fuel efficiency  

## ✅ Learning Outcomes  

By completing this notebook, you will learn how to:  

- Load and inspect real-world datasets  
- Handle missing data effectively  
- Perform descriptive statistics with Pandas  
- Use NumPy arrays for numerical operations  
- Apply filtering and transformations for analysis  
