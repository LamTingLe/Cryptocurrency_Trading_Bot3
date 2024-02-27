import datetime

import numpy as np
import pandas as pd  # pandas
import plotly.graph_objects as go  # plotly
import streamlit as st  # streamlit
import trendln
from sklearn.linear_model import LinearRegression
from streamlit_autorefresh import st_autorefresh  # streamlit autorefresh
from stqdm import stqdm  # streamlit progress bar
import matplotlib.pyplot as plt  # matplotlib
import plotly.io as pio  # plotly theme

from data import update_crypto_data, preprocess_crypto_data
from metric import Evaluation
from price_prediction import find_best_lag, predict, lr_model_v2

import warnings

warnings.filterwarnings('ignore', 'Series.__getitem__')

st.set_page_config(layout='wide')  # set page layout to wide
st_autorefresh(interval=1000 * 60 * 60, limit=None, debounce=True, key=None)

cryptocurrencies_dict = {'BTC': 'Bitcoin', 'ETH': 'Ethereum', 'XRP': 'Ripple'}
horizon = 3
target = "close"

for key, value in cryptocurrencies_dict.items():
    temp_df = update_crypto_data(key)
    temp_df = preprocess_crypto_data(temp_df, key)
    st.header(value)
    st.subheader(f'Historical Price ({temp_df.index.min()} --- {temp_df.index.max()})')
    st.dataframe(temp_df)
    is_complete = pd.Series(temp_df.index == pd.date_range(start=temp_df.index.min(),
                                                           end=temp_df.index.max(),
                                                           freq=temp_df.index.freq)).all()
    st.write('Time series is complete: ', is_complete)

    backtest_pred = pd.DataFrame()
    metrics = pd.DataFrame()

    n_days = 31
    step = 24 // horizon

    end_date = temp_df.index[-1]
    # st.write(end_date)
    start_date = end_date - pd.Timedelta(days=n_days)
    # st.write(start_date)

    pred_date = f"{start_date.year}-{start_date.month}-{start_date.day} 09:00:00"
    pred_date = pd.to_datetime(pred_date)
    # st.write(pred_date)

    capital = 100000
    holding_crypto = False
    buy_price = 0
    stop_loss = 0.025  # 2.5% stop loss
    buy_points = []
    sell_points = []
    profit_gains = []

    for i in stqdm(range(n_days * step)):
        res = pd.DataFrame()
        metric = Evaluation()

        frame = temp_df.loc[:pred_date + pd.Timedelta(hours=horizon - 1)]
        # st.write(frame)
        train = frame.head(len(frame) - horizon)
        # st.write(train)
        y_test = frame.tail(horizon)[target]
        # st.write(y_test)

        # new model
        best_lag = find_best_lag(train, target)
        # st.write(best_lag)

        ## linear
        pred = lr_model_v2(frame, target, y_test.index, horizon, best_lag)
        # st.write(pred)
        metric.score("Linear Regression", y_test=y_test, y_pred=pred[target])
        # st.write(metric.metric)
        res = pd.concat([res, pred], axis=1)
        # st.write(res)

        res.columns = ["Linear Regression"]
        # st.write(res)

        # res['Mean'] = (res["Linear Regression"])
        # st.write(res)
        # metric.score("Mean", y_test=y_test, y_pred=res['Mean'])
        # st.write(metric.metric)

        backtest_pred = pd.concat([backtest_pred, res])
        # st.write(backtest_pred)

        metric = metric.show().reset_index().rename(columns={'index': 'Model'})
        # st.write(metric)
        metric.insert(0, 'Start', y_test.index.min())
        # st.write(metric)
        metric.insert(1, 'End', y_test.index.max())
        # st.write(metric)

        metrics = pd.concat([metrics, metric])
        # st.write(metrics)

        # Iterate over each hour
        for hour in range(horizon):
            # Update the frame DataFrame to the current hour
            frame = temp_df.loc[:pred_date + pd.Timedelta(hours=hour)]

            # Get the predicted price for the current hour
            predicted_price = pred.iloc[hour][target]

            # Get the actual price at the current hour
            current_price = frame.iloc[-1][target]

            # Get the predicted prices for the next few hours
            predicted_prices = pred.iloc[hour: hour + horizon][target] if hour < horizon - 1 else [current_price]

            # Calculate the slope of the predicted prices for the next 3 hours
            X = np.array(range(len(predicted_prices[:3]))).reshape(-1, 1)
            y = np.array(predicted_prices[:3]).reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)

            slope = model.coef_[0][0]

            # Find the index of the minimum predicted price within the next 3 hours
            min_price_index = np.argmin(predicted_prices[:3])

            # If the bot is not currently holding the cryptocurrency and the current hour is the same as the hour with the minimum
            # predicted price and the capital is greater than or equal to the current price and the slope is positive, buy it
            if not holding_crypto and hour == min_price_index and capital >= current_price and slope > 0:
                # Buy the cryptocurrency
                holding_crypto = True
                buy_price = current_price
                capital_after_buying = capital - buy_price
                st.write(
                    f"Time: {pred.index[hour]}, Current capital: {capital}, Buying at: {buy_price}, Capital after "
                    f"buying: {capital_after_buying}")
                capital = capital_after_buying
                # Append the current timestamp and buy price to buy_points
                buy_points.append((pred.index[hour], buy_price))

            # If the bot is holding the cryptocurrency and the current price is the maximum of the predicted prices
            # and is above the buy price, sell the cryptocurrency
            if holding_crypto and current_price == max(predicted_prices) and current_price > buy_price:
                # Sell the cryptocurrency
                sell_price = current_price
                capital_after_selling = capital + sell_price
                profit_gain = sell_price - buy_price
                st.write(
                    f"Time: {pred.index[hour]}, Current capital: {capital}, Selling at: {sell_price}, Capital after "
                    f"selling: {capital_after_selling}, Profit gain: {profit_gain}")
                holding_crypto = False
                capital = capital_after_selling
                # Append the current timestamp and sell price to sell_points
                sell_points.append((pred.index[hour], sell_price))

                # Append the current timestamp and profit_gain to profit_gains
                profit_gains.append((pred.index[hour], profit_gain))

            # If holding the cryptocurrency and current price is less than or equal to buy price * (1 - stop_loss),
            # sell it
            elif holding_crypto and current_price <= buy_price * (1 - stop_loss):
                # Sell the cryptocurrency
                sell_price = current_price
                capital_after_selling = capital + sell_price
                profit_gain = sell_price - buy_price
                st.write(
                    f"Time: {pred.index[hour]}, Current capital: {capital}, Selling at: {sell_price}, Capital after selling: {capital_after_selling}, Profit gain: {profit_gain}")
                holding_crypto = False
                capital = capital_after_selling
                # Append the current timestamp and sell price to sell_points
                sell_points.append((pred.index[hour], sell_price))

                # Append the current timestamp and profit_gain to profit_gains
                profit_gains.append((pred.index[hour], profit_gain))

        pred_date += pd.Timedelta(hours=horizon)
        # st.write(pred_date)
        # break

    actual = temp_df.loc[backtest_pred.index][[target]]

    # After the trading simulation loop
    df_buy_points = pd.DataFrame(buy_points, columns=['Buy Points', 'Buy Price'])
    df_sell_points = pd.DataFrame(sell_points, columns=['Sell Points', 'Sell Price'])
    df_profit_gains = pd.DataFrame(profit_gains, columns=['Timestamp', 'Profit Gain'])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual[target].values, mode='lines', name='Actual Price', line=dict(color='SkyBlue')))
    for col in backtest_pred.columns:
        fig.add_trace(go.Scatter(x=actual.index, y=backtest_pred[col].values, mode='lines', name=col, line=dict(color='LightCoral')))
    # Add buy_points and sell_points to the Plotly chart
    fig.add_trace(
        go.Scatter(x=df_buy_points['Buy Points'], y=df_buy_points['Buy Price'], mode='markers',
                   marker=dict(size=10, color='LightGreen', symbol='triangle-up'), name='Buy Points'))
    fig.add_trace(
        go.Scatter(x=df_sell_points['Sell Points'], y=df_sell_points['Sell Price'], mode='markers',
                   marker=dict(size=10, color='Red', symbol='triangle-down'), name='Sell Points'))
    # fig.add_trace(go.Scatter(x=temp_df.index, y=temp_df['close'], mode='lines', name='close'))
    fig.update_layout(title=f'{value} Historical Price', xaxis_title='Date', yaxis_title='Price (USD)',
                      legend_title='Legend', showlegend=True, autosize=True)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

    # Save the figure into an interactive HTML file
    pio.write_html(fig, 'buy_sell_simulation.html')

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df_profit_gains['Timestamp'], y=df_profit_gains['Profit Gain'], mode='lines', name='Profit Gain'))
    fig.update_layout(title='Profit Gain Over Time', xaxis_title='Timestamp', yaxis_title='Profit Gain (USD)',
                      legend_title='Legend', showlegend=True, autosize=True)
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)

    st.write(f"Final capital: {capital}")

    break

    # best_lag = get_best_lag(temp_df)
    # for lag in sorted([7 * 24, 24, best_lag], reverse=True):
    #     st.subheader(f'Trend in Past {lag} Hours')
    #
    #     current_frame = temp_df.tail(lag)['close']
    #
    #     fig = trendln.plot_sup_res_date(current_frame, current_frame.index, accuracy=8)  # why 8
    #
    #     plt.title(f'Trend in Past {lag} Hours')
    #     plt.xlabel('Date')
    #     plt.ylabel('Price (USD)')
    #
    #     fig.get_axes()[0].yaxis.tick_right()
    #
    #     st.pyplot(fig, use_container_width=False)
    #
    # now = datetime.datetime.now()
    # latest_datetime = temp_df.index.max()
    # horizons = [3, 6, 24]
    #
    # for horizon in horizons:
    #     file = f'{key}_{horizon}_hour_prediction.csv'
    #
    #     predicted_df = predict(temp_df, best_lag, horizon)
    #
    #     predicted_df.to_csv(file)
    #
    #     predicted_df = predicted_df["Mean"]
    #
    #     st.subheader(f'{horizon} Hours Ahead Prediction')
    #     fig = go.Figure()
    #     fig.add_trace(go.Scatter(x=predicted_df.index, y=predicted_df, mode='lines', name='Predicted Price'))
    #     fig.update_layout(title=f'{value} {horizon} Hours Ahead Prediction', xaxis_title='Date',
    #                       yaxis_title='Price (USD)', legend_title='Legend', showlegend=True, autosize=True)
    #     st.plotly_chart(fig, theme='streamlit', use_container_width=True)
    #     st.dataframe(predicted_df)
