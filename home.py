import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load data
stocks = pd.read_csv('stocks - STOCK (2).csv')
stocks['date'] = pd.to_datetime(stocks['date'])

# Prepare data
stocks = stocks.pivot_table(index='date', columns='Company', values='close', aggfunc='last').fillna(method='bfill')
returns = stocks.pct_change().dropna()
returns = returns.replace([np.inf, -np.inf], np.nan)

# Calculate expected returns and covariance matrix
expected_returns = returns.mean()
cov_matrix = returns.cov()

# Define optimization functions
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def min_volatility(weights):
    return portfolio_volatility(weights, cov_matrix)

# Streamlit UI
budget = st.number_input('Боломжтой хөрөнгө')
risk_level = st.number_input('Риск хязгаар:', min_value=0.0, max_value=1.0, value=0.5)

# Optimization
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Sum of weights constraint
               {'type': 'eq', 'fun': lambda x: portfolio_volatility(x, cov_matrix) - risk_level})  # Risk level constraint
bounds = tuple((0, 1) for _ in range(len(expected_returns)))
initial_guess = [1 / len(expected_returns) for _ in range(len(expected_returns))]  # Initial guess for optimizer

optimal_weights = minimize(min_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# Display results
if st.button('Тооцоолох'):
    st.write("Тохирох жин:")
    for company, weight in zip(stocks.columns, optimal_weights.x):
        st.write(f"{company}: {round(budget*weight, 2)}")
st.write("Түүх:")
for company in stocks.columns:
    st.write(f"### {company}")
    st.line_chart(stocks[company])
