from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


class Lag:
    def __init__(self, X_train, X_test, y_train, scaler):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.scaler = scaler

    def reshape(self, inplace=False):
        X_train, y_train = self.X_train.values, self.y_train.values
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        if inplace:
            self.X_train = X_train
            self.y_train = y_train

        return X_train, y_train


def create_lag(data, target, lags, reshape=False):
    # Use a simple regressor if required for lag creation
    # model = ForecasterAutoreg(regressor=LinearRegression(), lags=lags)
    model = ForecasterAutoreg(regressor=None, lags=lags)

    target_scaler = StandardScaler()
    train = data.copy()

    # Scale the target variable
    train[target] = target_scaler.fit_transform(train[[target]])

    # Create lagged features
    X_train, y_train = model.create_train_X_y(train[target])

    # Reorder columns if needed (verify this step)
    cols = X_train.columns
    X_train = X_train[cols[::-1]]
    X_train.columns = cols

    # Create the test data
    X_test = X_train.iloc[-1, 1:].to_frame().T
    X_test[f"lag{lags + 1}"] = y_train.iloc[-1]
    X_test.columns = cols

    # Create Lag object
    lag_data = Lag(X_train, X_test, y_train, target_scaler)

    if reshape:
        lag_data.reshape(inplace=True)

    return lag_data
