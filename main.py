import xarray as xr, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib, os, time, json

def flatten(var):
    return var.values.reshape(len(var.valid_time), -1)

def save_model(model, scaler, metadata, name):
    if not os.path.exists("multi_models"): os.makedirs("multi_models")
    if not os.path.exists("multi_metadata"): os.makedirs("multi_metadata")
    if not os.path.exists("multi_scalers"): os.makedirs("multi_scalers")
    
    joblib.dump(model, f"multi_models/{name}_model.pkl")
    joblib.dump(scaler, f"multi_scalers/{name}_scaler.pkl")
    with open(f"multi_metadata/{name}_meta.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"Model, scaler and metadata saved for {name}")

X_vars = {
    "sst": "Sea Surface Temperature",
    "t2m": "Temperature at 2m above ground",
    "u10": "Wind U component at 10m",
    "v10": "Wind V component at 10m",
    "msl": "Mean Sea Level Pressure",
}

activation_fns = ["relu", "logistic", "tanh", "identity"]
optimization_fns = ["adam", "sgd", "lbfgs"]

print("Loading dataset...")
ds = xr.open_dataset("data_2014-2024.nc")
ds = ds.sel(latitude=slice(-55, -75))

X_vars = ["sst", "t2m", "u10", "v10", "msl"]
X_list = [flatten(ds[v]) for v in X_vars]

X = np.hstack(X_list)
Y = flatten(ds["siconc"])

# Cleaning data
print("Cleaning up the data...")

X = X[:, ~np.isnan(X).all(axis=0)]
Y = Y[:, ~np.isnan(Y).all(axis=0)]

X = SimpleImputer(strategy='mean').fit_transform(X)
Y = np.nan_to_num(Y, nan=0.0)

# Train-test split
print("Train-test splitting...")

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, shuffle=False
)

# Standardizing data
print("Standardizing the data...")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# MLP Model parameter ranges

hidden_layers = []
activation = activation_fns
optimization = optimization_fns
alpha = 0.001
max_iter = 200

n_layers = 1
print('Note: Enter "done" when finished adding layers')

while True:
    range_n_neurons = input(f"Enter range of neurons in layer {n_layers}: ")
    if range_n_neurons.lower() == "done":
        break
    try:
        start, end = map(int, range_n_neurons.split(","))
        hidden_layers.append((start, end))
        n_layers += 1
    except:
        print("Invalid input. Please enter two integers separated by a comma.")

print("Creating MLP model...")
model = MLPRegressor(
    hidden_layer_sizes=hidden_layers,
    activation=activation,
    solver=optimization,
    learning_rate_init=alpha,
    max_iter=max_iter,
    verbose=True
)

# Train
print("Training model...")
t0 = time.time()
model.fit(X_train, Y_train)
tf = time.time() - t0
print(f"Training completed in {tf:.2f} seconds")
print("Number of training epochs: " + str(model.n_iter_))
print("Final training loss: " + str(model.loss_))

# Predict
print("Running predictions...")
pred = np.clip(model.predict(X_test), 0, 1)

# Metrics
rmse = np.sqrt(mean_squared_error(Y_test, pred))
mae = mean_absolute_error(Y_test, pred)
corr = np.corrcoef(Y_test.flatten(), pred.flatten())[0, 1]

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"Correlation: {corr}")

metadata = {
    "variables": X_vars,
    "hidden_layers": hidden_layers,
    "activation": activation,
    "solver": optimization,
    "learning_rate": alpha,
    "max_iter": max_iter,
    "epochs": model.n_iter_,
    "rmse": rmse,
    "mae": mae,
    "corr": corr,
    "training_time": tf,
    "timestamp": time.time(),
}

# Save model
model_name = f"mlp_{hidden_layers}_{activation}_{optimization}_{alpha:.4f}_{model.n_iter_}"
save_model(model, scaler, metadata, model_name)
print(f"Saved model as: {model_name}")