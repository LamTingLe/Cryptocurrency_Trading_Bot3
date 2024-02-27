import streamlit as st
import plotly.graph_objects as go

import pandas as pd


def moving_average_strategy(df, window_size):
    # Calculate the moving average
    df['SMA'] = df['close'].rolling(window=window_size).mean()

    # Create a column 'Signal' such that if the closing price is greater than SMA then 1 else 0
    df['Signal'] = 0.0
    df.loc[df['close'] > df['SMA'], 'Signal'] = 1.0

    # Create a column 'Positions' which is the difference of the 'Signal' column
    df['Positions'] = df['Signal'].diff()

    # Create a column 'Net' to record the net between each buy and sell
    df['Net'] = df['close'].diff().where(df['Positions'] != 0)

    # Create a column 'Cumulative Net' to record the cumulative sum of 'Net'
    df['Cumulative Net'] = df['Net'].cumsum()

    # Forward fill the NaN values in 'Cumulative Net'
    df['Cumulative Net'] = df['Cumulative Net'].ffill()

    return df


def plot_signals(df):
    # Plot the closing price and moving average
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA'], mode='lines', name='Moving Average'))

    # Add markers for the buy and sell signals
    fig.add_trace(
        go.Scatter(x=df[df['Positions'] == 1].index, y=df[df['Positions'] == 1]['close'], mode='markers', name='Buy',
                   marker=dict(color='green', size=10, symbol='triangle-up')))
    fig.add_trace(
        go.Scatter(x=df[df['Positions'] == -1].index, y=df[df['Positions'] == -1]['close'], mode='markers', name='Sell',
                   marker=dict(color='red', size=10, symbol='triangle-down')))

    # Show the plot
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)


def plot_net(df):
    # Plot the net between each buy and sell
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Net'], mode='lines', name='Net'))

    # Add the cumulative sum line
    fig.add_trace(go.Scatter(x=df.index, y=df['Cumulative Net'], mode='lines', name='Cumulative Net'))

    # Show the plot
    st.plotly_chart(fig, theme='streamlit', use_container_width=True)
