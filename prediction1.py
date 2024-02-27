from sklearn.linear_model import LinearRegression

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf

from lag import create_lag


def find_best_lag(data, target):
    acf_val, conf = acf(data[target], nlags=len(data) - 1, alpha=.05)
    lower = conf[1:, 0] - acf_val[1:]
    upper = conf[1:, 1] - acf_val[1:]

    acf_res = pd.DataFrame({"lag": np.arange(1, len(acf_val)), "acf": acf_val[1:], "upper": upper, "lower": lower})
    acf_res["significant"] = acf_res.apply(lambda row: row['acf'] > 0 and row['acf'] > row.upper and row.lag > 1,
                                           axis=1)

    best_lag = acf_res[acf_res.significant]
    best_lag = best_lag.tail(1).lag.values[0]

    return int(best_lag)


def lr_model_v2(data, target, test_date, forecast, best_lag):
    lag_data = create_lag(data, target, best_lag, reshape=False)

    scaler = lag_data.scaler

    y_pred = []
    for hour in range(forecast):
        y_train = lag_data.y_train.shift(-hour).dropna()
        X_train = lag_data.X_train.loc[y_train.index]

        X_test = lag_data.X_test.iloc[[0]]

        model = LinearRegression()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        pred = scaler.inverse_transform([pred])[0][0]
        y_pred.append(pred)

    return pd.DataFrame({target: y_pred}, index=test_date)
