import yfinance as yf

def fetch_data(tickers, start_date, end_date):
  
  """Downloads historical data for given tickers and dates."""
  
  try:
    print("Fetching data...")
    data = yf.download(tickers, start=start_date, end=end_date)
    return data
  except Exception as e:
    print(f"Error downloading data: {e}")
    return None