import xarray as xr, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer
import joblib, os, time, json, itertools

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

X_vars_dict = {
    "sst": "Sea Surface Temperature",
    "t2m": "Temperature at 2m above ground",
    "u10": "Wind U component at 10m",
    "v10": "Wind V component at 10m",
    "msl": "Mean Sea Level Pressure",
}

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

# Hidden Layer Neuron Ranges
neuron_ranges = []
n_layers = 1
print('\n=== Layer Configuration ===')
print('Enter ranges (e.g., "4,8") to test all sizes between 4 and 8.')
print('Type "done" when you have defined all layers.')

while True:
    range_input = input(f"Enter range for Layer {n_layers} (start,end): ")
    if range_input.lower() == "done":
        break
    try:
        start, end = map(int, range_input.split(","))
        neuron_ranges.append(range(start, end + 1))
        n_layers += 1
    except ValueError:
        print("Invalid input. Please enter two integers separated by a comma (e.g., 4,8).")

hidden_layer_combinations = list(itertools.product(*neuron_ranges))

print(f"\nGenerated {len(hidden_layer_combinations)} combinations to test.")

# Parameter Grid
param_grid = {
    'hidden_layer_sizes': hidden_layer_combinations,
    'activation': ["relu", "tanh", "logistic"],
    'solver': ["adam", "sgd"],
    'learning_rate_init': [0.001, 0.005, 0.01],
    'max_iter': [200]
}

print("\nStarting Grid Search...")
print("Note: This may take a while :)")

mlp = MLPRegressor(random_state=1869)
grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    cv=3,       # 3-fold cross-validation
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1   # use all CPU cores
)

t0 = time.time()
grid_search.fit(X_train, Y_train)
tf = time.time() - t0

# Best Model Evaluation
print(f"\nGrid Search completed in {tf:.2f} seconds")
print(f"Best Parameters found: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
print("Running predictions on Test Set with best model...")
pred = np.clip(best_model.predict(X_test), 0, 1)

# Metrics
rmse = np.sqrt(mean_squared_error(Y_test, pred))
mae = mean_absolute_error(Y_test, pred)
corr = np.corrcoef(Y_test.flatten(), pred.flatten())[0, 1]

print(f"Best Model RMSE: {rmse}")
print(f"Best Model MAE: {mae}")
print(f"Best Model Correlation: {corr}")

metadata = {
    "variables": X_vars,
    "best_params": grid_search.best_params_,
    "search_space_size": len(hidden_layer_combinations) * len(param_grid['activation']) * len(param_grid['solver']),
    "rmse": rmse,
    "mae": mae,
    "corr": corr,
    "training_time": tf,
    "timestamp": time.time(),
}

# Save best model
model_name = f"best_mlp_{int(time.time())}"
save_model(best_model, scaler, metadata, model_name)
print(f"Saved best model as: {model_name}")