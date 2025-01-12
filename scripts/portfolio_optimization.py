import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate):
  
  """
  Calculate portfolio performance: return, volatility, and Sharpe ratio.
  """

  portfolio_return = np.dot(weights, mean_returns) * 252  # Annualized
  portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)  # Annualized
  sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
  return portfolio_return, portfolio_volatility, sharpe_ratio


def minimize_volatility(weights, mean_returns, cov_matrix):
  
  """
  Objective function to minimize portfolio volatility.
  
  """
  return portfolio_performance(weights, mean_returns, cov_matrix)[1]  # Return volatility


def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
  
  """
  Objective function to maximize Sharpe ratio (negative because we minimize).
  """
  
  return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]


def efficient_frontier(mean_returns, cov_matrix, risk_free_rate, num_points=100, allow_short_selling=False):
  
  """
  Generate points for the efficient frontier.
  """
  
  frontier_returns = np.linspace(min(mean_returns) * 252, max(mean_returns) * 252, num_points)
  frontier_volatilities = []
  for ret in frontier_returns:
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Sum of weights = 1
    if not allow_short_selling:
      bounds = tuple((0, 1) for _ in range(len(mean_returns)))  # No short selling
    else:
      bounds = None  # Allow short selling
    constraints_with_return = constraints + ({'type': 'eq', 'fun': lambda w: np.dot(w, mean_returns) * 252 - ret},)
    result = minimize(
      minimize_volatility,
      x0=np.array([1 / len(mean_returns)] * len(mean_returns)),
      args=(mean_returns, cov_matrix),
      method='SLSQP',
      bounds=bounds,
      constraints=constraints_with_return
    )
    if result.success:
      frontier_volatilities.append(result.fun)
    else:
      frontier_volatilities.append(np.nan)
  return frontier_returns, frontier_volatilities


def plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, optimal_weights_sharpe, optimal_weights_volatility, asset_names=None):
  
  """
  Plot the efficient frontier with optimal portfolios highlighted.
  """
  
  frontier_returns, frontier_volatilities = efficient_frontier(mean_returns, cov_matrix, risk_free_rate)
  
  plt.figure(figsize=(12, 6))
  plt.plot(frontier_volatilities, frontier_returns, 'b--', label="Efficient Frontier")
  plt.scatter(
    portfolio_performance(optimal_weights_sharpe, mean_returns, cov_matrix)[1],
    portfolio_performance(optimal_weights_sharpe, mean_returns, cov_matrix)[0],
    color='r', label='Max Sharpe Ratio Portfolio'
    )
  plt.scatter(
    portfolio_performance(optimal_weights_volatility, mean_returns, cov_matrix)[1],
    portfolio_performance(optimal_weights_volatility, mean_returns, cov_matrix)[0],
    color='g', label='Min Volatility Portfolio'
    )
  plt.title("Efficient Frontier")
  plt.xlabel("Portfolio Volatility (Annualized)")
  plt.ylabel("Portfolio Return (Annualized)")
  plt.legend()
  plt.grid()
  plt.show()