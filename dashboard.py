# import streamlit as st
# from scripts.fetch_data import fetch_data
# from scripts.preprocessing import clean_data, calculate_returns, visualize_data, print_summary
# from scripts.eda import load_data, analyze_returns, analyze_volatility, analyze_cumulative_returns
# from scripts.portfolio_analysis import calculate_var, calculate_cvar, calculate_drawdown, visualize_drawdown, visualize_var_cvar
# from scripts.portfolio_optimization import portfolio_performance, minimize_volatility, negative_sharpe_ratio, efficient_frontier, plot_efficient_frontier
# from scipy.optimize import minimize
# import pandas as pd
# import numpy as np
# import datetime
# import matplotlib.pyplot as plt

# st.set_page_config(page_title="Portfolio Analysis Dashboard", layout="wide")

# # Streamlit App
# st.title("Financial Portfolio Analysis Dashboard")

# # Data Upload Section
# uploaded_close_prices_file = st.file_uploader("Upload Historical Close Prices CSV")
# uploaded_returns_file = st.file_uploader("Upload Daily Returns CSV")

# # Risk-Free Rate & Confidence Level Input
# risk_free_rate = st.number_input("Annualized Risk-Free Rate (e.g., 0.02 for 2%)", min_value=0.0, max_value=1.0, step=0.001) / 252
# confidence_level = st.number_input("Confidence Level (e.g., 0.95 for 95%)", min_value=0.0, max_value=1.0, step=0.01)

# if uploaded_close_prices_file is not None and uploaded_returns_file is not None:
#   close_prices_data = pd.read_csv(uploaded_close_prices_file)
#   returns_data = pd.read_csv(uploaded_returns_file)

#   # Data Summary
#   st.header("Data Summary")
#   st.write("**Close Prices**")
#   st.dataframe(close_prices_data.describe())
#   st.write("**Daily Returns**")
#   st.dataframe(returns_data.describe())

#   # Risk Analysis
#   st.header("Risk Analysis")
#   analyze_returns(returns_data)
#   analyze_volatility(returns_data)
#   analyze_cumulative_returns(returns_data)

#   # VaR & CVaR Analysis
#   Var = calculate_var(returns_data, confidence_level)
#   cVar = calculate_cvar(returns_data, confidence_level)
#   visualize_drawdown(calculate_drawdown(returns_data))
#   visualize_var_cvar(returns_data, Var, cVar, confidence_level)

#   # Portfolio Optimization
#   mean_returns = returns_data.mean()
#   cov_matrix = returns_data.cov()
#   num_assets = len(mean_returns)
#   initial_weights = np.array([1.0 / num_assets] * num_assets)

#   constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Sum of weights = 1
#   bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling

#   optimal_sharpe = minimize(
#       negative_sharpe_ratio,
#       x0=initial_weights,
#       args=(mean_returns, cov_matrix, risk_free_rate),
#       method='SLSQP',
#       bounds=bounds,
#       constraints=constraints
#   )
#   optimal_weights_sharpe = optimal_sharpe.x

#   optimal_volatility = minimize(
#       minimize_volatility,
#       x0=initial_weights,
#       args=(mean_returns, cov_matrix),
#       method='SLSQP',
#       bounds=bounds,
#       constraints=constraints
#   )
#   optimal_weights_volatility = optimal_volatility.x

#   # Calculate Portfolio Performance
#   portfolio_return_sharpe, portfolio_volatility_sharpe, portfolio_sharpe_ratio_sharpe = portfolio_performance(optimal_weights_sharpe, mean_returns, cov_matrix, risk_free_rate)
#   portfolio_return_volatility, portfolio_volatility_volatility, portfolio_sharpe_ratio_volatility = portfolio_performance(optimal_weights_volatility, mean_returns, cov_matrix, risk_free_rate)

#   # Display Results
#   st.header("Portfolio Optimization Results")
#   st.write("**Optimal Weights for Max Sharpe Ratio:**")
#   st.write(pd.DataFrame(optimal_weights_sharpe, index=returns_data.columns, columns=['Weight']))
#   st.write("**Portfolio Performance (Max Sharpe Ratio):**")
#   st.write(f"Expected Return: {portfolio_return_sharpe:.2f}%")
#   st.write(f"Volatility: {portfolio_volatility_sharpe:.2f}")
#   st.write(f"Sharpe Ratio: {portfolio_sharpe_ratio_sharpe:.2f}")

#   st.write("\n**Optimal Weights for Min Volatility:**")
#   st.write(pd.DataFrame(optimal_weights_volatility, index=returns_data.columns, columns=['Weight']))
#   st.write("**Portfolio Performance (Min Volatility):**")
#   st.write(f"Expected Return: {portfolio_return_volatility:.2f}%")
#   st.write(f"Volatility: {portfolio_volatility_volatility:.2f}")
#   st.write(f"Sharpe Ratio: {portfolio_sharpe_ratio_volatility:.2f}")

#   # Efficient Frontier
#   st.header("Efficient Frontier")
#   frontier_returns, frontier_volatilities = efficient_frontier(mean_returns, cov_matrix, risk_free_rate)
#   plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, optimal_weights_sharpe, optimal_weights_volatility, frontier_returns, frontier_volatilities)
#   st.pyplot(plt)

# if __name__ == "__main__":
#   st.set_page_config(page_title="Portfolio Analysis Dashboard", layout="wide")

import streamlit as st
from scripts.fetch_data import fetch_data
from scripts.preprocessing import clean_data, calculate_returns, visualize_data, print_summary
from scripts.eda import load_data, analyze_returns, analyze_volatility, analyze_cumulative_returns
from scripts.portfolio_analysis import calculate_var, calculate_cvar, calculate_drawdown, visualize_drawdown, visualize_var_cvar
from scripts.portfolio_optimization import portfolio_performance, minimize_volatility, negative_sharpe_ratio, efficient_frontier, plot_efficient_frontier
from scipy.optimize import minimize
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

st.set_page_config(page_title="Portfolio Analysis Dashboard", layout="wide")

st.title("Financial Portfolio Analysis Dashboard")

# Data Upload Section
uploaded_close_prices_file = st.file_uploader("Upload Historical Close Prices CSV")
uploaded_returns_file = st.file_uploader("Upload Daily Returns CSV")

# Risk-Free Rate & Confidence Level Input
risk_free_rate = st.number_input("Annualized Risk-Free Rate (e.g., 0.02 for 2%)", min_value=0.0, max_value=1.0, step=0.001) / 252
confidence_level = st.number_input("Confidence Level (e.g., 0.95 for 95%)", min_value=0.0, max_value=1.0, step=0.01)

if uploaded_close_prices_file is not None and uploaded_returns_file is not None:
    close_prices_data = pd.read_csv(uploaded_close_prices_file)
    returns_data = pd.read_csv(uploaded_returns_file)

    returns_data_without_date = returns_data.drop('Date', axis=1)

    # Data Summary
    st.header("Data Summary")

    # Create two columns for displaying data side by side
    col1, col2 = st.columns(2)

    # Display Close Prices in the first column
    with col1:
        st.subheader("Close Prices Summary")
        st.dataframe(close_prices_data.describe())

    # Display Daily Returns in the second column
    with col2:
        st.subheader("Daily Returns Summary")
        st.dataframe(returns_data.describe())
    

    # Risk Analysis
    st.header("Risk Analysis")
    col3, col4, col5 = st.columns(3)

    with col3:
        fig_1 = analyze_returns(returns_data)  # This line now includes the visualization
        st.pyplot(fig_1)
        plt.close(fig_1)
    
    with col4:
        fig_2 = analyze_volatility(returns_data)  # This line now includes the visualization
        st.pyplot(fig_2)
        plt.close(fig_2)
    
    with col5:
        fig_3 = analyze_cumulative_returns(returns_data_without_date)  # This line now includes the visualization
        st.pyplot(fig_3)
        plt.close(fig_3)

    # VaR & CVaR Analysis
    Var = calculate_var(returns_data_without_date, confidence_level)
    cVar = calculate_cvar(returns_data_without_date, confidence_level)

    col6, col7 = st.columns(2)
    with col6:
        fig_4 = visualize_drawdown(calculate_drawdown(returns_data_without_date))
        st.pyplot(fig_4)
        plt.close(fig_4)
    
    with col7:
        fig_5 = visualize_var_cvar(returns_data_without_date, Var, cVar, confidence_level)
        st.pyplot(fig_5)
        plt.close(fig_5)

    # Portfolio Optimization
    mean_returns = returns_data_without_date.mean()
    cov_matrix = returns_data_without_date.cov()
    num_assets = len(mean_returns)
    initial_weights = np.array([1.0 / num_assets] * num_assets)

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # Sum of weights = 1
    bounds = tuple((0, 1) for _ in range(num_assets))  # No short selling

    optimal_sharpe = minimize(
        negative_sharpe_ratio,
        x0=initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    optimal_weights_sharpe = optimal_sharpe.x

    optimal_volatility = minimize(
        minimize_volatility,
        x0=initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    optimal_weights_volatility = optimal_volatility.x

    # Calculate Portfolio Performance
    portfolio_return_sharpe, portfolio_volatility_sharpe, portfolio_sharpe_ratio_sharpe = portfolio_performance(optimal_weights_sharpe, mean_returns, cov_matrix, risk_free_rate)
    portfolio_return_volatility, portfolio_volatility_volatility, portfolio_sharpe_ratio_volatility = portfolio_performance(optimal_weights_volatility, mean_returns, cov_matrix, risk_free_rate)

    # Display Results
    st.header("Portfolio Optimization Results")
    col8, col9 = st.columns(2)
    with col8:
        st.write("**Optimal Weights for Max Sharpe Ratio:**")
        st.write(pd.DataFrame(optimal_weights_sharpe, index=returns_data_without_date.columns, columns=['Weight']))
        st.write("**Portfolio Performance (Max Sharpe Ratio):**")
        st.write(f"Expected Return: {portfolio_return_sharpe:.2f}%")
        st.write(f"Volatility: {portfolio_volatility_sharpe:.2f}")
        st.write(f"Sharpe Ratio: {portfolio_sharpe_ratio_sharpe:.2f}")

    with col9:
        st.write("\n**Optimal Weights for Min Volatility:**")
        st.write(pd.DataFrame(optimal_weights_volatility, index=returns_data_without_date.columns, columns=['Weight']))
        st.write("**Portfolio Performance (Min Volatility):**")
        st.write(f"Expected Return: {portfolio_return_volatility:.2f}%")
        st.write(f"Volatility: {portfolio_volatility_volatility:.2f}")
        st.write(f"Sharpe Ratio: {portfolio_sharpe_ratio_volatility:.2f}")

    # Efficient Frontier
    st.header("Efficient Frontier")
    frontier_returns, frontier_volatilities = efficient_frontier(mean_returns, cov_matrix, risk_free_rate)
    fig_6 = plot_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, optimal_weights_sharpe, optimal_weights_volatility, frontier_returns, frontier_volatilities)
    st.pyplot(fig_6)
    plt.close(fig_6)

# if __name__ == "__main__":
#     st.set_page_config(page_title="Portfolio Analysis Dashboard", layout="wide")