# House Price Predictor

This project predicts house prices based on features like size, number of bedrooms, and age using **Linear Regression**.

## Dataset

The dataset contains 15 records with the following columns:
- `Size` (in square meters)
- `Bedrooms` (number of bedrooms)
- `Age` (in years)
- `Price` (in USD)

## What It Does

1. Loads housing data from a CSV file
2. Trains a linear regression model
3. Evaluates accuracy using Mean Squared Error
4. Plots actual vs predicted prices
5. Shows a custom prediction example

## Example Output Mean Squared Error: 123456.78
## Predicted price for a 2200 sq.m, 3 bed, 10 yr old house: $475,000.00


## Requirements

- Python 3.10+
- `pandas`
- `matplotlib`
- `scikit-learn`

Install with:

```bash
pip install pandas matplotlib scikit-learn
