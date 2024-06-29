import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import streamlit as st


class TradingBot:
    def __init__(self):
        self.initial_capital = 1000000
        self.capital = self.initial_capital
        self.holding_crypto = False
        self.buy_price = 0
        self.buy_points = []
        self.sell_points = []
        self.net_profit_gains = []
        self.profit_gains = []
        self.total_trades = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.capital_over_time = []
        self.timestamp = None
        self.price_change = None
        self.price_slope = None
        self.win_rate_over_time = []
        self.trade_durations = []
        self.buy_timestamp = None
        self.num_buys = 0
        self.num_sells = 0

        # Initialize the trades DataFrame
        self.trades = pd.DataFrame(columns=['Buy Timestamp', 'Buy Price', 'Sell Timestamp', 'Sell Price',
                                            'Profit', 'Return Rate'])

    def calculate_price_change(self, current_price):
        if self.buy_price:
            self.price_change = current_price - self.buy_price
        return self.price_change

    def calculate_price_slope(self, current_price, previous_price):
        if previous_price:
            self.price_slope = (current_price - previous_price) / previous_price
        return self.price_slope

    def decide_trade_prediction(self, current_price, future_price, timestamp):
        if not self.holding_crypto and future_price > current_price:
            self.buy(current_price, timestamp)
        elif self.holding_crypto and future_price < current_price:
            self.sell(current_price, timestamp)

    def decide_trade_momentum(self, current_price, future_price, previous_price, timestamp):
        if not self.holding_crypto and future_price > current_price > previous_price:
            self.buy(current_price, timestamp)
        elif self.holding_crypto and future_price < current_price < previous_price:
            self.sell(current_price, timestamp)

    def decide_trade_trend(self, current_price, future_prices, timestamp):
        # Fit a line to the predicted prices
        model = LinearRegression()
        model.fit(np.arange(len(future_prices)).reshape(-1, 1), future_prices)

        # Calculate the slope of the line
        slope = model.coef_[0]

        # Decide whether to buy or sell based on the slope
        if not self.holding_crypto and slope > 0:
            self.buy(current_price, timestamp)
        elif self.holding_crypto and slope < 0:
            self.sell(current_price, timestamp)

    def buy(self, current_price, timestamp):
        if current_price > self.capital:
            st.write("Insufficient capital to buy at the current price.")
            return
        self.timestamp = timestamp
        self.buy_price = current_price
        self.buy_points.append((timestamp, current_price))  # Record the buy point
        self.buy_timestamp = timestamp
        self.holding_crypto = True  # Update holding_crypto
        self.num_buys += 1  # Increment the buy counter
        self.capital -= current_price  # Deduct the buy price from the capital

        # Add a new row to the trades DataFrame for the buy operation
        new_trade = pd.DataFrame({'Buy Timestamp': [timestamp], 'Buy Price': [current_price]})
        self.trades = pd.concat([self.trades.dropna(axis=1, how='all'), new_trade], ignore_index=True)

    def sell(self, current_price, timestamp):
        if not self.holding_crypto:
            st.write("No cryptocurrency to sell.")
            return
        self.timestamp = timestamp
        self.calculate_price_change(current_price)
        self.sell_points.append((timestamp, current_price))  # Record the sell point
        self.trade_durations.append(timestamp - self.buy_timestamp)

        # Calculate profit
        profit = current_price - self.buy_price
        self.net_profit_gains.append(profit)  # Renamed from profit_gains

        return_rate = (profit / self.buy_price) * 100 if self.buy_price != 0 else 0

        # Update the last row of the trades DataFrame for the sell operation
        self.trades.at[self.trades.index[-1], 'Sell Timestamp'] = timestamp
        self.trades.at[self.trades.index[-1], 'Sell Price'] = current_price
        self.trades.at[self.trades.index[-1], 'Profit'] = profit
        self.trades.at[self.trades.index[-1], 'Return Rate'] = return_rate

        # Update trade counts
        self.total_trades += 1
        if profit > 0:
            self.successful_trades += 1
            self.profit_gains.append(profit)
        else:
            self.failed_trades += 1
        self.win_rate_over_time.append(self.calculate_win_rate())
        self.holding_crypto = False  # Update holding_crypto
        self.num_sells += 1  # Increment the sell counter
        self.capital += current_price  # Add the sell price to the capital

    def calculate_total_profit(self):
        self.total_profit = sum(self.net_profit_gains)
        return self.total_profit

    def calculate_win_rate(self):
        return self.successful_trades / self.total_trades if self.total_trades > 0 else 0

    def get_successful_trades(self):
        return self.successful_trades

    def get_failed_trades(self):
        return self.failed_trades

    def get_initial_capital(self):
        return self.initial_capital

    def get_final_capital(self):
        final_capital = self.capital
        if self.holding_crypto:
            final_capital += self.buy_price
        return final_capital

    def get_num_buys(self):
        return self.num_buys

    def get_num_sells(self):
        return self.num_sells

    def get_total_trades(self):
        return self.total_trades

    def average_profit_per_trade(self):
        return self.calculate_total_profit() / self.total_trades if self.total_trades > 0 else 0

    def average_profit_per_successful_trade(self):
        if self.profit_gains:
            avg_profit = sum(self.profit_gains) / len(self.profit_gains)
            return avg_profit
        else:
            return 0

    def average_trade_duration(self):
        if self.trade_durations:
            avg_duration = sum(self.trade_durations, datetime.timedelta()) / len(self.trade_durations)
            return str(avg_duration)
        else:
            return "0"

    def calculate_roi(self):
        net_profit = self.get_final_capital() - self.get_initial_capital()
        roi = (net_profit / self.get_initial_capital()) * 100
        return roi

    def calculate_roi_per_trade(self):
        if not self.buy_price:
            return 0
        net_profit_per_trade = self.sell_points[-1][1] - self.buy_price if self.sell_points else 0
        roi_per_trade = (net_profit_per_trade / self.buy_price) * 100
        return roi_per_trade

    def calculate_average_return_rate(self):
        if not self.trades.empty:
            return self.trades['Return Rate'].mean()
        else:
            return 0

    # def update_metrics(self):
    #     # Update the metrics here
    #
    # def plot_metrics(self):
    #     # Implement the plotting logic here
