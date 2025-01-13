import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_file):
  
  """Loads data from a CSV file, handling potential errors.

  Args:
      data_file (str): Path to the CSV file containing financial data.

  Returns:
      pandas.DataFrame: The loaded DataFrame, or None if an error occurs.
  """

  try:
    data = pd.read_csv(data_file, index_col='Date', parse_dates=True)
    return data
  except FileNotFoundError:
    print(f"Error: Data file not found - '{data_file}'.")
    return None


def analyze_returns(returns_data):
  
  """Analyzes and visualizes daily returns data.

  Args:
      returns_data (pandas.DataFrame): DataFrame containing daily returns.
  """

  # Step 3: Visualize Daily Returns
  plt.figure(figsize=(12, 6))
  for ticker in returns_data.columns:
    plt.plot(returns_data.index, returns_data[ticker], label=ticker)
  plt.title("Daily Returns Over Time")
  plt.xlabel("Date")
  plt.ylabel("Returns")
  plt.legend(loc="upper left")
  plt.grid()
  plt.show()

  # Step 4: Histogram of Returns
  plt.figure(figsize=(12, 6))
  for ticker in returns_data.columns:
    sns.histplot(returns_data[ticker], kde=True, label=ticker, bins=50, alpha=0.6)
  plt.title("Distribution of Daily Returns")
  plt.xlabel("Returns")
  plt.ylabel("Frequency")
  plt.legend(loc="upper left")
  plt.show()

  # Step 5: Correlation Matrix
  plt.figure(figsize=(10, 8))
  returns_data_without_date = returns_data.drop('Date', axis=1)
  sns.heatmap(returns_data_without_date.corr(), annot=True, cmap="coolwarm", fmt=".2f")
  plt.title("Correlation Matrix of Asset Returns")
  plt.show()
  return plt.gcf()


def analyze_volatility(returns_data):
  
  """Analyzes and visualizes volatility of asset returns.

  Args:
      returns_data (pandas.DataFrame): DataFrame containing daily returns.
  """

  # Step 6: Analyze Volatility
  returns_data_without_date = returns_data.drop('Date', axis=1)
  volatility = returns_data_without_date.std() * np.sqrt(252)  # Annualized volatility
  print("\nAnnualized Volatility (Std Dev):")
  print(volatility)

  # Step 7: Rolling Volatility
  plt.figure(figsize=(12, 6))
  for ticker in returns_data.columns:
    if pd.api.types.is_numeric_dtype(returns_data[ticker]):
      rolling_vol = returns_data[ticker].rolling(window=30).std() * np.sqrt(252)
    # rolling_vol = returns_data[ticker].rolling(window=30).std() * np.sqrt(252)
      plt.plot(rolling_vol, label=f"{ticker} (30-Day)")
  plt.title("Rolling Annualized Volatility (30-Day)")
  plt.xlabel("Date")
  plt.ylabel("Volatility")
  plt.legend(loc="upper left")
  plt.grid()
  plt.show()
  return plt.gcf()


def analyze_cumulative_returns(returns_data):
  
  """Analyzes and visualizes cumulative returns of assets.

  Args:
      returns_data (pandas.DataFrame): DataFrame containing daily returns.
  """

  # Step 8: Cumulative Returns
  cumulative_returns = (1 + returns_data).cumprod()
  plt.figure(figsize=(12, 6))
  cumulative_returns.plot()
  plt.title("Cumulative Returns Over Time")
  plt.xlabel("Date")
  plt.ylabel("Cumulative Returns")
  plt.grid()
  plt.show()
  return plt.gcf()
