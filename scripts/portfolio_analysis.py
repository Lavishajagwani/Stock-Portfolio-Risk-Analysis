import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_var(returns, confidence_level):
  
  """Calculates Value at Risk (VaR) for a given confidence level.

  Args:
      returns (pandas.DataFrame): DataFrame containing daily returns.
      confidence_level (float): Confidence level (e.g., 0.95).

  Returns:
      pandas.Series: VaR for each asset in the portfolio.
  """
  var = returns.quantile(1 - confidence_level)
  return var


def calculate_cvar(returns, confidence_level):
  
  """Calculates Conditional Value at Risk (CVaR) for a given confidence level.

  Args:
      returns (pandas.DataFrame): DataFrame containing daily returns.
      confidence_level (float): Confidence level (e.g., 0.95).

  Returns:
      pandas.Series: CVaR for each asset in the portfolio.
  """
  var = calculate_var(returns, confidence_level)
  cvar = returns[returns < var].mean()
  return cvar


# def calculate_sharpe_ratio(returns, risk_free_rate):
  
#   """Calculates the Sharpe Ratio.

#   Args:
#       returns (pandas.DataFrame): DataFrame containing daily returns.
#       risk_free_rate (float): Risk-free rate (e.g., 0.02).

#   Returns:
#       float: Annualized Sharpe Ratio.
#   """
#   sharpe_ratio = (returns.mean() - risk_free_rate) / returns.std()
#   annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)
#   return annualized_sharpe_ratio


def calculate_drawdown(returns):
  
  """Calculates the maximum drawdown of the portfolio.

  Args:
      returns (pandas.DataFrame): DataFrame containing daily returns.

  Returns:
      float: Maximum drawdown.
  """
  cumulative_returns = (1 + returns).cumprod()
  rolling_max = cumulative_returns.cummax()
  drawdown = (cumulative_returns - rolling_max) / rolling_max
  return drawdown
#   max_drawdown = drawdown.min()
#   return max_drawdown


def visualize_drawdown(drawdown):
  
  """Visualizes the drawdown of the portfolio over time.

  Args:
      drawdown (pandas.Series): Series containing drawdown values.
  """
  plt.figure(figsize=(12, 6))
  drawdown.plot(title="Drawdown Over Time", ylabel="Drawdown", xlabel="Date")
  plt.axhline(0, color='red', linestyle='--', alpha=0.7)
  plt.grid()
  plt.show()


def visualize_var_cvar(returns, var, cvar, confidence_level):
  
  """Visualizes VaR and CVaR for each asset in the portfolio.

  Args:
      returns (pandas.DataFrame): DataFrame containing daily returns.
      var (pandas.Series): VaR for each asset.
      cvar (pandas.Series): CVaR for each asset.
      confidence_level (float): Confidence level.
  """
  plt.figure(figsize=(12, 6))
  for ticker in returns.columns:
    sns.histplot(returns[ticker], bins=50, kde=True, label=ticker, alpha=0.6)
    plt.axvline(var[ticker], color='red', linestyle='--', label=f"VaR ({ticker})")
    plt.axvline(cvar[ticker], color='blue', linestyle='--', label=f"CVaR ({ticker})")
  plt.title(f"Value at Risk (VaR) and Conditional VaR ({confidence_level*100:.0f}%)")
  plt.xlabel("Returns")
  plt.ylabel("Frequency")
  plt.legend()
  plt.grid()
  plt.show()