import dearpygui.dearpygui as dpg
import xarray as xr, numpy as np, time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import SimpleImputer

def log(msg):
    dpg.set_value("log_window", dpg.get_value("log_window") + msg + "\n")

def flatten(var):
    return var.values.reshape(len(var.valid_time), -1)

X_vars = {
    "sst": "Sea Surface Temperature",
    "t2m": "Temperature at 2m above ground",
    "u10": "Wind U component at 10m",
    "v10": "Wind V component at 10m",
    "msl": "Mean Sea Level Pressure",
}

activation_fns = {
    "Rectified Linear Unit": "relu",
    "Hyperbolic Tangent": "tanh",
    "Sigmoidal Function": "logistic",
}

optimization_fns = {
    "Adam (Adaptive Moment Estimation)": "adam",
    "Stochastic Gradient Descent": "sgd",
    "Limited-memory BFGS": "lbfgs",
}

log("Loading dataset...")
ds = xr.open_dataset("data_2014-2024.nc")

def run_training():
    try:
        dpg.set_value("log_window", "")

        selected_vars = []
        for var in X_vars.keys():
            if dpg.get_value(f"chk_{var}"):
                selected_vars.append(var)

        if len(selected_vars) == 0:
            log("ERROR: Select at least one X variable.")
            return

        log(f"Selected variables: {selected_vars}")

        X_list = [flatten(ds[v]) for v in selected_vars]
        X = np.hstack(X_list)
        Y = flatten(ds["siconc"])

        log("Cleaning up the data...")

        # Remove NaN columns
        X = X[:, ~np.isnan(X).all(axis=0)]
        Y = Y[:, ~np.isnan(Y).all(axis=0)]

        # Impute and fill NaNs
        X = SimpleImputer(strategy='mean').fit_transform(X)
        Y = np.nan_to_num(Y, nan=0.0)

        # Train-test split
        log("Train-test splitting...")
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=dpg.get_value("ttsplit"), shuffle=False
        )

        # Standardizing
        log("Standardizing the data...")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # Read (hyper)parameters from GUI
        layers_str = dpg.get_value("lst_hidden_layers")
        hidden_layers = tuple(map(int, layers_str.split(",")))

        activation = activation_fns[dpg.get_value("activation_combo")]
        optimization = optimization_fns[dpg.get_value("optimization_combo")]
        alpha = dpg.get_value("alpha")
        max_iter = dpg.get_value("max_iters")

        # Create model
        log("Creating MLP model...")
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=optimization,
            learning_rate_init=alpha,
            max_iter=max_iter,
            verbose=True
        )

        # Train
        log("Training model...")
        t0 = time.time()
        model.fit(X_train, Y_train)
        log(f"Training completed in {time.time() - t0:.2f} seconds")
        log("Number of training epochs: " + str(model.n_iter_))
        log("Final training loss: " + str(model.loss_))

        # Predict
        log("Running predictions...")
        pred = np.clip(model.predict(X_test), 0, 1)
        # pred = model.predict(X_test)
        # log(f"First 5 predicted sea-ice conc: {pred[:5]}")
        # log(f"First 5 actual sea-ice conc: {Y_test[:5]}")

        # Metrics
        rmse = np.sqrt(mean_squared_error(Y_test, pred))
        mae = mean_absolute_error(Y_test, pred)
        corr = np.corrcoef(Y_test.flatten(), pred.flatten())[0, 1]

        log(f"RMSE: {rmse}")
        log(f"MAE: {mae}")
        log(f"Correlation: {corr}")

        dpg.set_value("rmse_val", f"{rmse:.4f}")
        dpg.set_value("mae_val", f"{mae:.4f}")
        dpg.set_value("corr_val", f"{corr:.4f}")

    except BaseException as e:
        log(f"ERROR:\n{e}")

dpg.create_context()

with dpg.window(tag="Sea-Ice Prediction MLP"):
    dpg.add_text("Select input variables (X):")
    for var in X_vars.keys():
        dpg.add_checkbox(label=X_vars[var], default_value=True, tag=f"chk_{var}")

    dpg.add_input_float(label="Testing split size",default_value=0.3, tag="ttsplit")

    dpg.add_separator()

    dpg.add_text("Model (hyper)parameters:")
    dpg.add_input_text(label="Hidden layer perceptrons (comma separated)", default_value="5,5,5", tag="lst_hidden_layers")
    dpg.add_combo(list(activation_fns.keys()), label="Activation", default_value=list(activation_fns.values())[0], tag="activation_combo")
    dpg.add_combo(list(optimization_fns.keys()), label="Optimization", default_value=list(optimization_fns.values())[0], tag="optimization_combo")
    dpg.add_input_float(label="Learning Rate (alpha)", default_value=0.001, tag="alpha", step=0.001)
    dpg.add_input_int(label="Max Iterations", default_value=200, tag="max_iters")

    dpg.add_separator()
    dpg.add_button(label="Train Model", callback=run_training)
    dpg.add_separator()

    dpg.add_text("Results:")
    dpg.add_text("RMSE:", bullet=True); dpg.add_same_line(); dpg.add_text("", tag="rmse_val")
    dpg.add_text("MAE:", bullet=True); dpg.add_same_line(); dpg.add_text("", tag="mae_val")
    dpg.add_text("Correlation:", bullet=True); dpg.add_same_line(); dpg.add_text("", tag="corr_val")

    dpg.add_text("Logs:")
    dpg.add_input_text(tag="log_window", multiline=True, width=500, height=200, readonly=True)

dpg.create_viewport(title="Sea-Ice Prediction MLP")
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Sea-Ice Prediction MLP", True)
dpg.start_dearpygui()
dpg.destroy_context()