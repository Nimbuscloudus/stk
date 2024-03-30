import streamlit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from pypfopt import expected_returns, risk_models, EfficientFrontier
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rates = pd.read_csv('exchange rates - Currency (2).csv')
stocks = pd.read_csv('stocks - STOCK (2).csv')

rates.date = rates.date.astype('datetime64[ns]')
stocks.date = stocks.date.astype('datetime64[ns]')
stocks = stocks.pivot_table(index='date', columns='Company', values='close', aggfunc='last')
stocks = stocks.fillna(method='bfill')
returns = stocks.pct_change().dropna()
returns = returns.replace([np.inf, -np.inf], np.nan)
expected_returns = returns.mean()
cov_matrix = returns.cov()
def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def min_volatility(weights):
    return portfolio_volatility(weights, cov_matrix)

constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0, 1) for _ in range(len(expected_returns)))

initial_guess = [1 / len(expected_returns) for _ in range(len(expected_returns))]

optimal_weights = minimize(min_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

print("Тохирох жин:")
for company, weight in zip(stocks.columns, optimal_weights.x):
    print(f"{company}: {weight:.4f}")
