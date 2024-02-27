import datetime
import os

import pandas as pd
from cryptocompare import cryptocompare


# function to update cryptocurrency data hourly and save it to csv file
def update_crypto_data(cryptocurrency):
    now = datetime.datetime.now()
    filename = f'{cryptocurrency}_hour_raw.csv'

    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        temp_df = df.copy()
        temp_df['datetime'] = pd.to_datetime(temp_df['time'], unit='s') + pd.Timedelta(hours=8)
        latest_datetime = temp_df['datetime'].max() + pd.Timedelta(hours=1)

        if latest_datetime >= now:
            return df

    data = cryptocompare.get_historical_price_hour_from(cryptocurrency, 'USD', exchange='CCCAGG',
                                                        toTs=now,
                                                        fromTs=datetime.datetime(2023, 1, 1),
                                                        delay=0.2)

    # Create the directory if it does not exist
    os.makedirs('cryptocurrency_data', exist_ok=True)

    filename = f'cryptocurrency_data/{cryptocurrency}_hour_raw.csv'
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df


def preprocess_crypto_data(df, cryptocurrency):
    df['datetime'] = pd.to_datetime(df['time'], unit='s') + pd.Timedelta(hours=8)

    df.set_index('datetime', inplace=True)
    df = df.asfreq('h')
    df.sort_index(inplace=True)

    df = df[['high', 'low', 'open', 'volumefrom', 'volumeto', 'close', 'conversionType', 'conversionSymbol']]

    # Create the directory if it does not exist
    os.makedirs('cryptocurrency_data', exist_ok=True)

    filename = f'cryptocurrency_data/{cryptocurrency}_hour_preprocess.csv'
    df.to_csv(filename)
    return df
