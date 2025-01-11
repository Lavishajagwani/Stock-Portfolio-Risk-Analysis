import matplotlib.pyplot as plt
import seaborn as sns

def clean_data(data):
  
  """Handles missing values in the data."""
  
  data['Close'].fillna(method='ffill', inplace=True)
  data['Close'].dropna(inplace=True)
  return data


def calculate_returns(data):
  
  """Calculates daily percentage returns for each asset."""
  
  returns = data['Close'].pct_change().dropna()
  return returns


def visualize_data(data, returns):
 
  """Creates visualizations for historical prices and asset correlation."""
  
  plt.figure(figsize=(12, 6))

  # Plot for each ticker
  for ticker in data.columns:
    data[ticker].plot(label=ticker)

  plt.title("Historical Close Prices")
  plt.xlabel("Date")
  plt.ylabel("Close Price")
  plt.grid()
  plt.show()


def print_summary(data, tickers, start_date, end_date):
  
  """Prints a summary of the downloaded data."""
  
  print("\nData Summary:")
  print(f"Start Date: {start_date}, End Date: {end_date}")
  print(f"Number of Assets: {len(tickers)}")  # Adjust for other columns in yf.download()
  print(f"Dataset Shape: {data.shape}")

