import xarray as xr
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import time

def flatten(var):
    return var.values.reshape(len(var.valid_time), -1)

ds = xr.open_dataset("data_2014-2024.nc")
X_vars = ["sst", "t2m", "u10", "v10", "msl"]
X_list = [flatten(ds[v]) for v in X_vars]

X = np.hstack(X_list)
Y = flatten(ds["siconc"])

X = X[:, ~np.isnan(X).all(axis=0)]
Y = Y[:, ~np.isnan(Y).all(axis=0)]

X = SimpleImputer(strategy='mean').fit_transform(X)
Y = np.nan_to_num(Y, nan=0.0)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, shuffle=False
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=200,
    learning_rate_init=0.001,
    verbose=True
)

t0 = time.time()
model.fit(X_train, Y_train)
print("Training time:", time.time() - t0, "seconds")

pred = model.predict(X_test)
pred = np.clip(pred, 0, 1)

print("RMSE:", np.sqrt(mean_squared_error(Y_test, pred)))
print("MAE:", mean_absolute_error(Y_test, pred))