import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import sys

report_folder = os.path.join('reports')
os.makedirs(report_folder, exist_ok=True)
output_file_path = os.path.join(report_folder, 'exploratory_analysis_report.txt')
sys.stdout = open(output_file_path, 'w', encoding='utf-8')

warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Load the dataset
print("=" * 80)
print("SUPERSTORE DATA - EXPLORATORY DATA ANALYSIS")
print("=" * 80)
print(f"\nLoading cleaned data...")

input_file_path = os.path.join('data', 'processed', 'Superstore_Cleaned.csv')
if not os.path.exists(input_file_path):
    raise FileNotFoundError(
        f'Cleaned dataset not found: {input_file_path}.\n'
        'Please run the preprocessing script first or place the cleaned CSV in the data/processed folder.'
    )

try:
    data = pd.read_csv(input_file_path, encoding='utf-8')
except UnicodeDecodeError:
    data = pd.read_csv(input_file_path, encoding='latin-1')

print(f"\n✓ Data loaded successfully from {input_file_path}!")
print(f"Dataset shape: {data.shape[0]} rows × {data.shape[1]} columns")

# ============================================================================
# 1. BASIC DATASET OVERVIEW
# ============================================================================
print("\n" + "=" * 80)
print("1. DATASET OVERVIEW")
print("=" * 80)

print("\nFirst few rows:")
print(data.head())

print("\nData Types:")
print(data.dtypes)

print("\nMissing Values:")
missing = data.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values detected!")
else:
    print(missing[missing > 0])

print("\nDataset Info:")
print(data.info())

# ============================================================================
# 2. STATISTICAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("2. STATISTICAL SUMMARY")
print("=" * 80)

print("\nNumeric Columns Summary:")
print(data.describe())

print("\nCategorical Columns Summary:")
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"\n{col}:")
    print(f"  Unique values: {data[col].nunique()}")
    print(f"  Top 5 values:\n{data[col].value_counts().head()}")

# ============================================================================
# 3. SALES ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("3. SALES ANALYSIS")
print("=" * 80)

total_sales = data['Sales'].sum()
total_orders = data['Order ID'].nunique()
total_customers = data['Customer ID'].nunique()
avg_order_value = total_sales / total_orders

print(f"\nTotal Sales: ${total_sales:,.2f}")
print(f"Total Orders: {total_orders:,}")
print(f"Total Customers: {total_customers:,}")
print(f"Average Order Value: ${avg_order_value:,.2f}")

# Sales by Category
print("\n\nSales by Category:")
sales_by_category = data.groupby('Category')['Sales'].agg(['sum', 'mean', 'count'])
sales_by_category['percentage'] = (sales_by_category['sum'] / total_sales * 100).round(2)
print(sales_by_category)

# Sales by Segment
print("\n\nSales by Segment:")
sales_by_segment = data.groupby('Segment')['Sales'].agg(['sum', 'mean', 'count'])
sales_by_segment['percentage'] = (sales_by_segment['sum'] / total_sales * 100).round(2)
print(sales_by_segment)

# Sales by Region
print("\n\nSales by Region:")
sales_by_region = data.groupby('Region')['Sales'].agg(['sum', 'mean', 'count'])
sales_by_region['percentage'] = (sales_by_region['sum'] / total_sales * 100).round(2)
print(sales_by_region)

# ============================================================================
# 4. PROFITABILITY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("4. PROFITABILITY ANALYSIS")
print("=" * 80)

total_profit = data['Profit'].sum()
profit_margin = (total_profit / total_sales) * 100
loss_orders = (data['Profit'] < 0).sum()
loss_percentage = (loss_orders / len(data)) * 100

print(f"\nTotal Profit: ${total_profit:,.2f}")
print(f"Profit Margin: {profit_margin:.2f}%")
print(f"Orders with Loss: {loss_orders} ({loss_percentage:.2f}%)")

# Profit by Category
print("\n\nProfit by Category:")
profit_by_category = data.groupby('Category')['Profit'].agg(['sum', 'mean'])
profit_by_category['margin%'] = (data.groupby('Category').apply(
    lambda x: (x['Profit'].sum() / x['Sales'].sum() * 100)
))
print(profit_by_category)

# Loss-making Products (Top 10)
print("\n\nTop 10 Loss-Making Products:")
loss_products = data[data['Profit'] < 0].groupby('Product Name').agg({
    'Profit': 'sum',
    'Sales': 'count',
    'Discount': 'mean'
}).sort_values('Profit')
print(loss_products.head(10))

# Discount Impact
print("\n\nDiscount Impact on Profit:")
print(f"Average Discount: {data['Discount'].mean():.2%}")
print(f"Orders with Discount: {(data['Discount'] > 0).sum()} ({(data['Discount'] > 0).sum()/len(data)*100:.2f}%)")

# ============================================================================
# 5. CUSTOMER ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("5. CUSTOMER ANALYSIS")
print("=" * 80)

# Customer Distribution
print("\n\nCustomer Distribution by Segment:")
customer_segment = data.groupby('Segment')['Customer ID'].nunique()
print(customer_segment)

# Customer Orders
print("\n\nOrders per Customer Statistics:")
orders_per_customer = data.groupby('Customer ID').size()
print(f"  Min Orders: {orders_per_customer.min()}")
print(f"  Max Orders: {orders_per_customer.max()}")
print(f"  Avg Orders: {orders_per_customer.mean():.2f}")
print(f"  Repeat Customers (2+ orders): {(orders_per_customer >= 2).sum()} ({(orders_per_customer >= 2).sum()/total_customers*100:.2f}%)")

# Top Customers
print("\n\nTop 10 Customers by Profit:")
top_customers = data.groupby('Customer Name').agg({
    'Profit': 'sum',
    'Sales': 'sum',
    'Order ID': 'nunique'
}).sort_values('Profit', ascending=False)
top_customers.columns = ['Total_Profit', 'Total_Sales', 'Orders']
print(top_customers.head(10))

# ============================================================================
# 6. TEMPORAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. TEMPORAL ANALYSIS")
print("=" * 80)

# Convert dates
data['Order Date'] = pd.to_datetime(data['Order Date'])
data['Ship Date'] = pd.to_datetime(data['Ship Date'])

# Date range
print(f"\nDate Range:")
print(f"  Earliest Order: {data['Order Date'].min().date()}")
print(f"  Latest Order: {data['Order Date'].max().date()}")
print(f"  Time Span: {(data['Order Date'].max() - data['Order Date'].min()).days} days")

# Shipping Time
data['Days to Ship'] = (data['Ship Date'] - data['Order Date']).dt.days
print(f"\n\nShipping Time Analysis:")
print(f"  Average Days to Ship: {data['Days to Ship'].mean():.2f}")
print(f"  Min Days: {data['Days to Ship'].min()}")
print(f"  Max Days: {data['Days to Ship'].max()}")
print(f"  Median Days: {data['Days to Ship'].median():.0f}")

# Orders by Ship Mode
print("\n\nOrders by Ship Mode:")
ship_mode_stats = data.groupby('Ship Mode').agg({
    'Order ID': 'count',
    'Sales': 'sum',
    'Profit': 'sum',
    'Days to Ship': 'mean'
})
ship_mode_stats.columns = ['Orders', 'Sales', 'Profit', 'Avg_Days']
print(ship_mode_stats)

# ============================================================================
# 7. GEOGRAPHIC ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("7. GEOGRAPHIC ANALYSIS")
print("=" * 80)

# States with most orders
print("\nTop 10 States by Sales:")
top_states = data.groupby('State')['Sales'].sum().sort_values(ascending=False).head(10)
print(top_states)

# Regional breakdown
print("\n\nRegional Breakdown:")
regional_stats = data.groupby('Region').agg({
    'Order ID': 'nunique',
    'Customer ID': 'nunique',
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
})
regional_stats.columns = ['Orders', 'Customers', 'Sales', 'Profit', 'Quantity']
regional_stats['Margin%'] = (regional_stats['Profit'] / regional_stats['Sales'] * 100).round(2)
print(regional_stats)

# ============================================================================
# 8. PRODUCT ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("8. PRODUCT ANALYSIS")
print("=" * 80)

# Products by count
print(f"\nTotal Products: {data['Product Name'].nunique()}")
print(f"Total Sub-Categories: {data['Sub-Category'].nunique()}")

# Top Products by Sales
print("\n\nTop 10 Products by Sales:")
top_products_sales = data.groupby('Product Name').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum'
}).sort_values('Sales', ascending=False).head(10)
print(top_products_sales)

# Sub-category Analysis
print("\n\nSub-Category Analysis:")
subcat_stats = data.groupby('Sub-Category').agg({
    'Sales': 'sum',
    'Profit': 'sum',
    'Quantity': 'sum',
    'Order ID': 'count'
}).sort_values('Sales', ascending=False)
subcat_stats.columns = ['Sales', 'Profit', 'Quantity', 'Orders']
subcat_stats['Margin%'] = (subcat_stats['Profit'] / subcat_stats['Sales'] * 100).round(2)
print(subcat_stats)

# ============================================================================
# 9. DATA QUALITY FINDINGS
# ============================================================================
print("\n" + "=" * 80)
print("9. DATA QUALITY & FINDINGS")
print("=" * 80)

findings = []

# Check 1: Unusual Quantities
unusual_qty = (data['Quantity'] < 1).sum()
if unusual_qty > 0:
    findings.append(f"⚠ Found {unusual_qty} orders with quantity < 1")

# Check 2: Negative Prices
negative_sales = (data['Sales'] <= 0).sum()
if negative_sales > 0:
    findings.append(f"⚠ Found {negative_sales} orders with non-positive sales")

# Check 3: Profit Anomalies
loss_count = (data['Profit'] < 0).sum()
if loss_count > 0:
    findings.append(f"⚠ Found {loss_count} ({loss_count/len(data)*100:.1f}%) loss-making orders")

# Check 4: Discount Extremes
extreme_discount = (data['Discount'] > 0.5).sum()
if extreme_discount > 0:
    findings.append(f"⚠ Found {extreme_discount} orders with discount > 50%")

# Check 5: Ship Time Anomalies
negative_ship_time = (data['Days to Ship'] < 0).sum()
if negative_ship_time > 0:
    findings.append(f"⚠ Found {negative_ship_time} orders shipped before order date")

if findings:
    print("\nData Quality Issues Found:")
    for finding in findings:
        print(f"  {finding}")
else:
    print("\n✓ No major data quality issues detected!")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print("\nGenerated insights can be used for:")
print("  • PowerBI Dashboard creation")
print("  • Business strategy development")
print("  • Product/pricing optimization")
print("  • Customer retention initiatives")
print("  • Operational efficiency improvements")

sys.stdout.close()
sys.stdout = sys.__stdout__
print(f"Analysis output saved to: {output_file_path}")
