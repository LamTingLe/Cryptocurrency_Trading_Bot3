import datetime
import os
from datetime import timedelta

import numpy as np
import pandas as pd  # pandas
import plotly.graph_objects as go  # plotly
import streamlit as st  # streamlit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from streamlit_autorefresh import st_autorefresh  # streamlit autorefresh
from stqdm import stqdm  # streamlit progress bar
import plotly.io as pio  # plotly

from data import update_crypto_data, preprocess_crypto_data
from prediction1 import find_best_lag, lr_model_v2
from trading_bot import TradingBot
pio.json.config.default_engine = 'json'

title = 'Cryptocurrency Price Prediction and Trading Simulation'

st.set_page_config(page_title=title,
                   page_icon=':moneybag:',
                   layout='wide',
                   initial_sidebar_state='collapsed')  # set page layout to wide
st_autorefresh(interval=1000 * 60 * 60, limit=None, debounce=True, key=None)

cryptocurrencies_dict = {'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'XRP': 'Ripple'}
trading_strategies = ['Price Prediction Strategy', 'Momentum Strategy', 'Trend Analysis Strategy']
forecasts = [1, 3, 6, 12]
target = 'close'
prediction_folder = 'prediction_results'
trading_strategies_folder = 'trading_strategy_results'


def display_crypto_tabs():
    tabs = st.tabs([f"{value} ({key})" for key, value in cryptocurrencies_dict.items()])

    for i, tab in enumerate(tabs):
        with tab:
            key = list(cryptocurrencies_dict.keys())[i]
            display_and_plot_crypto_data(key, cryptocurrencies_dict[key])


def display_and_plot_crypto_data(key, value):
    st.header(f"{value} ({key})", anchor=False, divider='violet')
    temp_df = display_crypto_data(key)

    plot_crypto_data(temp_df, key)

    display_metric(temp_df, key, 3)

    filtered_df = select_and_filter_date_range(temp_df, key)

    n_days = forecast_dates(filtered_df, key)

    display_trading_strategy_tabs(filtered_df, key, n_days)


def display_crypto_data(key):
    df = update_crypto_data(key)
    df = preprocess_crypto_data(df, key)
    is_complete = pd.Series(df.index == pd.date_range(start=df.index.min(),
                                                      end=df.index.max(),
                                                      freq=df.index.freq)).all()
    with st.expander(f'{key}USD Historical Price ({df.index.min()} --- {df.index.max()})', expanded=False):
        st.write('Time series is complete: ', is_complete)
        st.dataframe(df, use_container_width=True, hide_index=False)

    return df


def plot_crypto_data(df, key):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='close'))
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='markers', name='close', visible='legendonly'))
    fig.update_layout(title=f'{key}USD Historical Close Price', xaxis_title='Date', yaxis_title='Price (USD)',
                      legend_title='Legend', showlegend=True, autosize=True)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)


def display_metric(df, label, hour):
    st.subheader(f'{label}USD Close Price (Past {hour} Hours)', anchor=False, divider='violet')
    cols = st.columns(hour, gap='small')
    for i in range(1, hour + 1):
        with cols[i - 1]:
            st.metric(label=f'{df.index[-i]}',
                      value=df['close'].iloc[-i],
                      delta=df['close'].iloc[-i] - df['close'].iloc[-i - 1],
                      delta_color='normal',
                      label_visibility='visible')


def select_and_filter_date_range(temp_df, key):
    st.subheader('Select Date Range', anchor=False, divider='violet')

    date_selection_method = st.radio(label='Select date range using:', options=('Slider', 'Date Input'), index=0,
                                     key=f'{key}_date_selection_method', horizontal=True, label_visibility='visible')

    min_date = temp_df.index.min()
    max_date = temp_df.index.max()

    if date_selection_method == 'Slider':
        min_date_slider = pd.to_datetime(min_date).date()
        max_date_slider = pd.to_datetime(max_date).date()

        date_range = st.slider(label='Select date range', min_value=min_date_slider, max_value=max_date_slider,
                               value=[datetime.date(2023, 1, 1),
                                      datetime.date(2023, 5, 31)],
                               step=timedelta(days=1), format='DD/MM/YYYY',
                               key=f'{key}_date_range_slider', label_visibility='visible')
    else:
        date_range = st.date_input(label='Select date range',
                                   value=[datetime.date(2023, 7, 1),
                                          datetime.date(2023, 11, 30)],
                                   min_value=min_date, max_value=max_date, key=f'{key}_date_range_input',
                                   format='DD/MM/YYYY', label_visibility='visible')

    if len(date_range) == 1:
        start_date = end_date = pd.to_datetime(date_range[0])
    else:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])

    filtered_df = temp_df[(temp_df.index >= start_date) & (temp_df.index <= end_date)]

    with st.expander(f"{start_date.strftime('%Y-%m-%d')} --- {end_date.strftime('%Y-%m-%d')}", expanded=False):
        plot_crypto_data(filtered_df, key)

    return filtered_df


def forecast_dates(filtered_df, key):
    st.subheader('Forecast Date Range', anchor=False, divider='violet')
    # Define end_date here
    end_date = filtered_df.index[-1]

    n_days = st.number_input(label='Number of Days to Forecast', min_value=1, max_value=180, value=90,
                             step=1, key=f'{key}_n_days',
                             format='%d', placeholder='Enter number of days', label_visibility='visible')

    # Display the dates involved in the forecast
    start_date = end_date - pd.Timedelta(days=n_days)

    # Filter the DataFrame based on the start_date and end_date
    filtered_df = filtered_df.loc[start_date:end_date]

    with st.expander(f"{start_date.strftime('%Y-%m-%d')} --- {end_date.strftime('%Y-%m-%d')} ", expanded=False):
        plot_crypto_data(filtered_df, key)

    return n_days


def display_trading_strategy_tabs(filtered_df, key, n_days):
    tabs = st.tabs(trading_strategies)

    for i, tab in enumerate(tabs):
        with tab:
            trading_strategy = trading_strategies[i]
            if trading_strategy == 'Price Prediction Strategy':
                display_forecast_tabs(filtered_df, key, n_days, trading_strategy)
            elif trading_strategy == 'Momentum Strategy':
                display_forecast_tabs(filtered_df, key, n_days, trading_strategy)
            elif trading_strategy == 'Trend Analysis Strategy':
                display_forecast_tabs(filtered_df, key, n_days, trading_strategy)


def run_forecast(filtered_df, step, n_days, forecast, start_date, csv_file):
    backtest_prediction = pd.DataFrame()
    for _ in stqdm(range(n_days * step), desc=f'Running {forecast} Hour Forecast'):
        results = pd.DataFrame()
        timeframe = filtered_df.loc[:start_date + pd.Timedelta(hours=forecast - 1)]
        train = timeframe.head(len(timeframe) - forecast)
        y_test = timeframe.tail(forecast)[target]

        best_lag = find_best_lag(train, target)

        predictions = lr_model_v2(timeframe, target, y_test.index, forecast, best_lag)
        predictions.columns = ['Linear Regression']
        results = pd.concat([results, predictions], axis=1)

        backtest_prediction = pd.concat([backtest_prediction, results])

        start_date += pd.Timedelta(hours=forecast)

    # Create the directory if it does not exist
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)

    backtest_prediction.to_csv(csv_file)
    return backtest_prediction


def load_backtest_prediction(csv_file):
    if not os.path.exists(csv_file):
        st.write("No backtest prediction available")
        return None
    else:
        return pd.read_csv(csv_file, index_col=0)


def perform_trading_operations(bot, actual, backtest_prediction, trading_strategy, forecast):
    previous_price = None
    for i in range(len(actual.index)):
        timestamp = actual.index[i]
        current_price = actual[target].values[i]
        future_price = backtest_prediction.values[i]
        if previous_price is not None:
            if trading_strategy == 'Price Prediction Strategy':
                bot.decide_trade_prediction(current_price, future_price, timestamp)
            elif trading_strategy == 'Momentum Strategy':
                bot.decide_trade_momentum(current_price, future_price, previous_price, timestamp)
            elif trading_strategy == 'Trend Analysis Strategy':
                # Check if there are enough future prices
                if i + forecast < len(backtest_prediction.values):
                    # Get the future prices for the next 'forecast' steps
                    future_prices = backtest_prediction.values[i + 1:i + 1 + forecast]
                    bot.decide_trade_trend(current_price, future_prices, timestamp)
        previous_price = current_price
    return bot


def plot_prediction_vs_actual(actual, backtest_prediction, key, forecast, bot):
    # Assume bot is your instance of TradingBot
    buy_points = bot.buy_points
    sell_points = bot.sell_points

    # Convert the points to a DataFrame for easier manipulation
    buy_df = pd.DataFrame(buy_points, columns=['timestamp', 'price'])
    sell_df = pd.DataFrame(sell_points, columns=['timestamp', 'price'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual[target].values, mode='lines', name='Actual Close Price',
                             line=dict(color='SkyBlue')))
    for col in backtest_prediction.columns:
        fig.add_trace(go.Scatter(x=actual.index, y=backtest_prediction[col].values, mode='lines', name=col,
                                 line=dict(color='LightCoral')))
    # Add traces for the buy and sell points
    fig.add_trace(go.Scatter(x=buy_df['timestamp'], y=buy_df['price'], mode='markers',
                             marker=dict(size=10, color='LightGreen', symbol='triangle-up'), name='Buy Points'))
    fig.add_trace(go.Scatter(x=sell_df['timestamp'], y=sell_df['price'], mode='markers',
                             marker=dict(size=10, color='Red', symbol='triangle-down'), name='Sell Points'))
    fig.update_layout(title=f'{key}USD Predicted Price vs Actual Price ({forecast} Hour Forecast)',
                      xaxis_title='Date', yaxis_title='Price (USD)',
                      legend_title='Legend', showlegend=True, autosize=True)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    return fig


def save_plot_files(fig, html_file, png_file):
    # Create the directories if they do not exist
    os.makedirs(os.path.dirname(html_file), exist_ok=True)
    os.makedirs(os.path.dirname(png_file), exist_ok=True)

    # Always write/overwrite the HTML and image files
    pio.write_html(fig, file=html_file, auto_open=False)
    pio.write_image(fig, file=png_file)


def display_forecast_tabs(filtered_df, key, n_days, trading_strategy):
    tabs = st.tabs([f"{forecast} Hour Forecast" for forecast in forecasts])

    for i, tab in enumerate(tabs):
        with (tab):
            forecast = forecasts[i]
            step = 24 // forecast

            end_date = filtered_df.index[-1]
            start_date = end_date - pd.Timedelta(days=n_days)

            csv_file = (f'{prediction_folder}/{key}/'
                        f'{key}_{forecast}h_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv')

            activate = st.button(label='Run Forecast', key=f'{key}_{forecast}_{trading_strategy}_activate_button',
                                 type='secondary', use_container_width=False)

            if activate:
                backtest_prediction = run_forecast(filtered_df, step, n_days, forecast, start_date, csv_file)
                activate = False
            else:
                backtest_prediction = load_backtest_prediction(csv_file)

            if backtest_prediction is None:
                continue

            actual = filtered_df.loc[backtest_prediction.index][[target]]

            # display_evaluation_metrics(actual, backtest_prediction)

            bot = TradingBot()

            bot = perform_trading_operations(bot, actual, backtest_prediction, trading_strategy, forecast)

            fig = plot_prediction_vs_actual(actual, backtest_prediction, key, forecast, bot)

            html_file = f'{trading_strategies_folder}/{trading_strategy}/{key}/' \
                        f'{key}_{forecast}h_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.html'
            png_file = f'{trading_strategies_folder}/{trading_strategy}/{key}/' \
                       f'{key}_{forecast}h_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.png'

            save_plot_files(fig, html_file, png_file)

            display_trading_metrics(bot)

            plot_all_graphs(bot)


def plot_all_graphs(bot):
    plot_trade_profit(bot)
    plot_cumulative_profit(bot)
    cols = st.columns(2, gap='small')
    with cols[0]:
        plot_win_rate(bot)
    with cols[1]:
        plot_trade_outcome(bot)
    plot_trade_duration(bot)
    plot_profit_loss_distribution(bot)


def plot_trade_profit(bot):
    # Plot the profit gained from each trade
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(bot.net_profit_gains))), y=bot.net_profit_gains, mode='lines'))
    fig.update_layout(title='Profit Gained from Each Trade', xaxis_title='Trade Number', yaxis_title='Profit (USD)')
    st.plotly_chart(fig, theme='streamlit', use_container_width=True, autosize=True)


def plot_cumulative_profit(bot):
    # Plot the cumulative profit over time
    cumulative_profit = np.cumsum(bot.net_profit_gains)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(cumulative_profit))), y=cumulative_profit, mode='lines'))
    fig.update_layout(title='Cumulative Profit Over Time', xaxis_title='Trade Number',
                      yaxis_title='Cumulative Profit (USD)')
    st.plotly_chart(fig, theme='streamlit', use_container_width=True, autosize=True)


def plot_win_rate(bot):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(bot.win_rate_over_time))), y=bot.win_rate_over_time, mode='lines'))
    fig.update_layout(title='Win Rate Over Time', xaxis_title='Trade Number', yaxis_title='Win Rate')
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)


def plot_trade_outcome(bot):
    fig = go.Figure(data=[go.Bar(x=['Successful Trades', 'Failed Trades'],
                                 y=[bot.get_successful_trades(), bot.get_failed_trades()])])
    fig.update_layout(title_text='Trade Outcome Distribution')
    st.plotly_chart(fig, use_container_width=True)


def plot_trade_duration(bot):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(bot.trade_durations))), y=bot.trade_durations, mode='lines'))
    fig.update_layout(title='Trade Duration', xaxis_title='Trade Number', yaxis_title='Duration (hours)')
    st.plotly_chart(fig, use_container_width=True)


def plot_profit_loss_distribution(bot):
    fig = go.Figure(data=[go.Histogram(x=bot.net_profit_gains)])
    fig.update_layout(title='Profit/Loss Distribution', xaxis_title='Profit/Loss', yaxis_title='Count')
    st.plotly_chart(fig, use_container_width=True)


# Define a function to calculate the metrics
def calculate_metrics(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** (1 / 2)
    r2 = r2_score(y_test, y_pred)
    return mae, mape, mse, rmse, r2


def display_evaluation_metrics(actual, backtest_prediction):
    # Calculate the evaluation metrics
    mae, mape, mse, rmse, r2 = calculate_metrics(actual, backtest_prediction)

    # Display the evaluation metrics using Streamlit cards
    st.subheader('Evaluation Metrics')
    cols = st.columns(5)
    with cols[0]:
        st.metric(label="Mean Absolute Error", value=f"{mae:.2f}")
    with cols[1]:
        st.metric(label="Mean Absolute Percentage Error", value=f"{mape:.2f}")
    with cols[2]:
        st.metric(label="Mean Squared Error", value=f"{mse:.2f}")
    with cols[3]:
        st.metric(label="Root Mean Squared Error", value=f"{rmse:.2f}")
    with cols[4]:
        st.metric(label="R2 Score", value=f"{r2:.2f}")


def display_trading_metrics(bot):
    st.subheader('Trading Metrics', anchor=False, divider='violet')

    total_profit = bot.calculate_total_profit()
    win_rate = bot.calculate_win_rate()
    successful_trades = bot.get_successful_trades()
    failed_trades = bot.get_failed_trades()
    total_trades = bot.get_total_trades()
    initial_capital = bot.initial_capital
    final_capital = bot.capital
    num_buys = bot.get_num_buys()
    num_sells = bot.get_num_sells()
    average_profit = bot.average_profit_per_trade()
    average_profit_per_successful_trade = bot.average_profit_per_successful_trade()
    average_trade_duration = bot.average_trade_duration()
    roi = bot.calculate_roi()

    cols = st.columns(3, gap='small')
    with cols[0]:
        st.metric(label="Initial Capital", value=initial_capital)
        st.metric(label="Total Trades", value=total_trades, delta_color='normal', label_visibility='visible')
        st.metric(label="Average Profit per Trade", value=average_profit,
                  delta_color='normal', label_visibility='visible')
        st.metric(label="Return on Investment (ROI)", value=roi, delta_color='normal', label_visibility='visible')
        st.metric(label="Number of Buys", value=num_buys, delta_color='normal', label_visibility='visible')

    with cols[1]:
        st.metric(label="Final Capital", value=final_capital, delta=final_capital - initial_capital,
                  delta_color='normal', label_visibility='visible')
        st.metric(label="Successful Trades", value=successful_trades, delta_color='normal', label_visibility='visible')
        st.metric(label="Average Profit per Successful Trade", value=average_profit_per_successful_trade,
                  delta_color='normal', label_visibility='visible')
        st.metric(label="Average Trade Duration", value=average_trade_duration, delta_color='normal',
                  label_visibility='visible')
        st.metric(label="Number of Sells", value=num_sells, delta_color='normal', label_visibility='visible')

    with cols[2]:
        st.metric(label="Total Profit", value=total_profit, delta_color='normal', label_visibility='visible')
        st.metric(label="Failed Trades", value=failed_trades, delta_color='normal', label_visibility='visible')
        st.metric(label="Win Rate", value=win_rate, delta_color='normal', label_visibility='visible')


st.title(title, anchor=False)

display_crypto_tabs()
