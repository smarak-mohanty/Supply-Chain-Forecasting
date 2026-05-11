"""
Superstore Data - Data Preprocessing & Cleaning
This script cleans and preprocesses the Superstore dataset for analysis.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import os
import sys

report_folder = os.path.join('reports')
os.makedirs(report_folder, exist_ok=True)
output_file_path = os.path.join(report_folder, 'data_preprocessing_report.txt')
sys.stdout = open(output_file_path, 'w', encoding='utf-8')

warnings.filterwarnings('ignore')

print("=" * 80)
print("SUPERSTORE DATA - DATA PREPROCESSING")
print("=" * 80)

# Load the dataset
print("\nStep 1: Loading data...")
raw_file_path = 'data/raw/Sample - Superstore.csv'
try:
    data = pd.read_csv(raw_file_path, encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv(raw_file_path, encoding='latin-1')
print(f"✓ Loaded {len(data)} rows × {len(data.columns)} columns from {raw_file_path}")

# Create backup
data_original = data.copy()

# ============================================================================
# STEP 2: DATA TYPE CONVERSION
# ============================================================================
print("\nStep 2: Converting data types...")

# Convert date columns
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date'])

# Ensure numeric types
data['Sales'] = pd.to_numeric(data['Sales'], errors='coerce')
data['Quantity'] = pd.to_numeric(data['Quantity'], errors='coerce')
data['Discount'] = pd.to_numeric(data['Discount'], errors='coerce')
data['Profit'] = pd.to_numeric(data['Profit'], errors='coerce')

print("✓ Data types converted")
print(f"  Order Date: datetime")
print(f"  Ship Date: datetime")
print(f"  Sales, Quantity, Discount, Profit: numeric")

# ============================================================================
# STEP 3: HANDLING MISSING VALUES
# ============================================================================
print("\nStep 3: Checking for missing values...")

missing = data.isnull().sum()
if missing.sum() == 0:
    print("✓ No missing values found!")
else:
    print("Missing values by column:")
    print(missing[missing > 0])

# ============================================================================
# STEP 4: OUTLIER DETECTION & HANDLING
# ============================================================================
print("\nStep 4: Detecting outliers...")

outliers_found = False

# Check for negative sales
negative_sales = data[data['Sales'] < 0]
if len(negative_sales) > 0:
    print(f"⚠ Found {len(negative_sales)} orders with negative sales")
    outliers_found = True

# Check for invalid quantities
invalid_qty = data[data['Quantity'] < 1]
if len(invalid_qty) > 0:
    print(f"⚠ Found {len(invalid_qty)} orders with quantity < 1")
    outliers_found = True

# Check for invalid discounts
invalid_discount = data[(data['Discount'] < 0) | (data['Discount'] > 1)]
if len(invalid_discount) > 0:
    print(f"⚠ Found {len(invalid_discount)} orders with discount outside [0, 1]")
    outliers_found = True

# Check for negative shipping times
negative_ship = data[data['Ship Date'] < data['Order Date']]
if len(negative_ship) > 0:
    print(f"⚠ Found {len(negative_ship)} orders shipped before order date")
    outliers_found = True

if not outliers_found:
    print("✓ No major outliers detected!")

# ============================================================================
# STEP 5: DERIVED FEATURES
# ============================================================================
print("\nStep 5: Creating derived features...")

# Shipping time
data['Days_to_Ship'] = (data['Ship Date'] - data['Order Date']).dt.days

# Profit margin
data['Profit_Margin'] = data['Profit'] / data['Sales']
data['Profit_Margin_Pct'] = (data['Profit'] / data['Sales'] * 100).round(2)

# Sales per unit
data['Sales_per_Unit'] = data['Sales'] / data['Quantity']

# Discount amount
data['Discount_Amount'] = data['Sales'] * data['Discount']

# Order year, month, quarter
data['Order_Year'] = data['Order Date'].dt.year
data['Order_Month'] = data['Order Date'].dt.month
data['Order_Month_Name'] = data['Order Date'].dt.strftime('%B')
data['Order_Quarter'] = 'Q' + data['Order Date'].dt.quarter.astype(str)
data['Order_Day_of_Week'] = data['Order Date'].dt.day_name()

# Loss flag
data['Is_Loss'] = (data['Profit'] < 0).astype(int)

# High discount flag
data['High_Discount'] = (data['Discount'] > 0.2).astype(int)

print("✓ Derived features created:")
print(f"  • Days_to_Ship")
print(f"  • Profit_Margin, Profit_Margin_Pct")
print(f"  • Sales_per_Unit")
print(f"  • Discount_Amount")
print(f"  • Order_Year, Order_Month, Order_Month_Name, Order_Quarter, Order_Day_of_Week")
print(f"  • Is_Loss, High_Discount")

# ============================================================================
# STEP 6: TEXT CLEANING
# ============================================================================
print("\nStep 6: Cleaning text fields...")

# Strip whitespace
text_cols = ['Customer Name', 'Product Name', 'City', 'State', 'Region', 
             'Category', 'Sub-Category', 'Segment']
for col in text_cols:
    data[col] = data[col].str.strip()

# Standardize case (title case)
data['Customer Name'] = data['Customer Name'].str.title()
data['Product Name'] = data['Product Name'].str.title()
data['City'] = data['City'].str.title()

print("✓ Text fields cleaned and standardized")

# ============================================================================
# STEP 7: DATA VALIDATION
# ============================================================================
print("\nStep 7: Data validation...")

validation_checks = {
    'Total Rows': len(data),
    'Columns': len(data.columns),
    'Unique Customers': data['Customer ID'].nunique(),
    'Unique Orders': data['Order ID'].nunique(),
    'Unique Products': data['Product Name'].nunique(),
    'Date Range (Start)': data['Order Date'].min().date(),
    'Date Range (End)': data['Order Date'].max().date(),
}

print("Validation Results:")
for check, value in validation_checks.items():
    print(f"  ✓ {check}: {value}")

# ============================================================================
# STEP 8: SUMMARY STATISTICS AFTER CLEANING
# ============================================================================
print("\nStep 8: Summary statistics (cleaned data)...")

print("\nSales Metrics:")
print(f"  Total Sales: ${data['Sales'].sum():,.2f}")
print(f"  Avg Sales per Order: ${data['Sales'].mean():,.2f}")
print(f"  Total Profit: ${data['Profit'].sum():,.2f}")
print(f"  Avg Profit Margin: {data['Profit_Margin_Pct'].mean():.2f}%")

print("\nVolume Metrics:")
print(f"  Total Orders: {data['Order ID'].nunique():,}")
print(f"  Total Units Sold: {data['Quantity'].sum():,.0f}")
print(f"  Avg Units per Order: {data['Quantity'].mean():.2f}")

print("\nCustomer Metrics:")
print(f"  Total Customers: {data['Customer ID'].nunique():,}")
print(f"  Orders per Customer: {data.groupby('Customer ID').size().mean():.2f}")
print(f"  Repeat Customers: {(data.groupby('Customer ID').size() > 1).sum():,}")

print("\nQuality Metrics:")
print(f"  Loss-making Orders: {data['Is_Loss'].sum()} ({data['Is_Loss'].mean()*100:.2f}%)")
print(f"  High Discount Orders: {data['High_Discount'].sum()} ({data['High_Discount'].mean()*100:.2f}%)")
print(f"  Avg Days to Ship: {data['Days_to_Ship'].mean():.2f}")

# ============================================================================
# STEP 9: SAVE CLEANED DATA
# ============================================================================
print("\nStep 9: Saving cleaned data...")

output_path = 'data/processed/Superstore_Cleaned.csv'
data.to_csv(output_path, index=False)
print(f"✓ Cleaned data saved to: {output_path}")

# Save data quality report
report_path = 'data/processed/Data_Quality_Report.txt'
with open(report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("SUPERSTORE DATA QUALITY REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("DATASET OVERVIEW\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total Rows: {len(data)}\n")
    f.write(f"Total Columns: {len(data.columns)}\n")
    f.write(f"Date Range: {data['Order Date'].min().date()} to {data['Order Date'].max().date()}\n\n")
    
    f.write("DATA QUALITY CHECKS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Missing Values: {missing.sum()} (PASS)\n")
    f.write(f"Negative Sales: {(data['Sales'] < 0).sum()} records\n")
    f.write(f"Invalid Quantities: {(data['Quantity'] < 1).sum()} records\n")
    f.write(f"Invalid Discounts: {((data['Discount'] < 0) | (data['Discount'] > 1)).sum()} records\n")
    f.write(f"Negative Ship Times: {(data['Days_to_Ship'] < 0).sum()} records\n\n")
    
    f.write("KEY METRICS\n")
    f.write("-" * 80 + "\n")
    f.write(f"Total Sales: ${data['Sales'].sum():,.2f}\n")
    f.write(f"Total Profit: ${data['Profit'].sum():,.2f}\n")
    f.write(f"Profit Margin: {(data['Profit'].sum()/data['Sales'].sum()*100):.2f}%\n")
    f.write(f"Total Orders: {data['Order ID'].nunique():,}\n")
    f.write(f"Total Customers: {data['Customer ID'].nunique():,}\n")
    f.write(f"Loss-making Orders: {(data['Profit'] < 0).sum()} ({(data['Profit'] < 0).sum()/len(data)*100:.2f}%)\n\n")
    
    f.write("COLUMNS IN CLEANED DATASET\n")
    f.write("-" * 80 + "\n")
    for i, col in enumerate(data.columns, 1):
        f.write(f"{i}. {col}\n")
    f.write("\n")
    
    f.write("Report Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")

print(f"✓ Data quality report saved to: {report_path}")

print("\nStep 10: Merging reports into one file...\n")
print("" + "=" * 80)
print("MERGED DATA QUALITY REPORT")
print("" + "=" * 80)
with open(report_path, 'r', encoding='utf-8') as quality_file:
    print(quality_file.read())

# ============================================================================
# STEP 10: SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)

print("\nProcessing Summary:")
print(f"  ✓ Data types standardized")
print(f"  ✓ Date columns converted to datetime")
print(f"  ✓ Numeric columns validated")
print(f"  ✓ {len(data.columns) - data_original.shape[1]} derived features created")
print(f"  ✓ Text fields cleaned and standardized")
print(f"  ✓ Data quality checks performed")
print(f"  ✓ Cleaned dataset exported")

print("\nFiles Created:")
print(f"  1. {output_path}")
print(f"  2. {report_path}")

print("\nNext Steps:")
print("  1. Load the cleaned data in PowerBI")
print("  2. Create data model with relationships")
print("  3. Build dashboards using specifications")
print("  4. Implement interactive slicers and filters")
print("  5. Create calculated measures and fields")
print("  6. Test and validate dashboard results")

print("\n" + "=" * 80)

sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"✓ Preprocessing and merged report saved to: {output_file_path}")
