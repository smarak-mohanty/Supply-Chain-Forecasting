# Supply Project

This project is a retail sales analytics pipeline built around the Superstore sample dataset.

## Project Overview

- Loads raw Superstore sales data from `data/raw/Sample - Superstore.csv`
- Cleans and preprocesses data in `analysis/data_preprocessing.py`
- Creates derived features for deeper analysis
- Saves cleaned output to `data/processed/Superstore_Cleaned.csv`
- Performs exploratory data analysis in `analysis/exploratory_analysis.py`
- Produces data quality and summary reports in `reports/`

## Data Processing

Key preprocessing steps:
- convert date columns to datetime
- convert numeric columns to appropriate numeric types
- detect missing values and outliers
- create derived features such as:
  - `Days_to_Ship`
  - `Profit_Margin`, `Profit_Margin_Pct`
  - `Sales_per_Unit`
  - `Discount_Amount`
  - `Order_Year`, `Order_Month`, `Order_Quarter`, `Order_Day_of_Week`
  - `Is_Loss`, `High_Discount`

## Analysis Goals

- Validate dataset quality and completeness
- Summarize sales, profit, order, and customer metrics
- Identify loss-making orders and discount behavior
- Support dashboard creation in Power BI or other BI tools

## Files and Structure

- `analysis/data_preprocessing.py` - preprocessing pipeline and data cleaning
- `analysis/exploratory_analysis.py` - exploratory data analysis and reporting
- `data/raw/` - raw source dataset
- `data/processed/` - cleaned dataset and processed output
- `reports/` - generated analysis reports
- `requirements.txt` - Python dependencies

## How to Use

1. Activate the virtual environment: `myenv\Scripts\Activate.ps1` (PowerShell) or `myenv\Scripts\activate.bat` (CMD)
2. Install dependencies: `pip install -r requirements.txt`
3. Run preprocessing: `python analysis/data_preprocessing.py`
4. Run exploratory analysis: `python analysis/exploratory_analysis.py`

## Notes

- The cleaned dataset is stored as `data/processed/Superstore_Cleaned.csv`
- Reports are saved in `reports/`
- The project is designed for dashboard-ready analytics and further BI visualization
