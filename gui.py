import dearpygui.dearpygui as dpg

X_vars = {
    "sst": "Sea Surface Temperature",
    "t2m": "Temperature at 2m above ground",
    "u10": "Wind U component at 10m",
    "v10": "Wind V component at 10m",
    "msl": "Mean Sea Level Pressure",
}

dpg.create_context()

with dpg.window(tag="Sea-Ice Prediction MLP"):
    dpg.add_text("Select input variables (X):")
    for var in X_vars.keys():
        dpg.add_checkbox(label=X_vars[var], default_value=True, tag=f"chk_{var}")

    dpg.add_separator()

    dpg.add_text("Model parameters:")
    dpg.add_input_text(label="Hidden layers (comma separated)", default_value="128,64", tag="lst_hidden_layers")
    dpg.add_combo(["relu", "tanh", "logistic"], label="Activation", default_value="relu", tag="activation_combo")
    dpg.add_combo(["adam", "sgd", "lbfgs"], label="Solver", default_value="adam", tag="solver_combo")
    dpg.add_input_float(label="Learning Rate", default_value=0.001, tag="alpha", step=0.001)
    dpg.add_input_int(label="Max Iterations", default_value=200, tag="num_iters")

    dpg.add_separator()
    dpg.add_button(label="Train Model")
    dpg.add_separator()

    dpg.add_text("Results:")
    dpg.add_text("RMSE:", bullet=True); dpg.add_same_line(); dpg.add_text("", tag="rmse_val")
    dpg.add_text("MAE:", bullet=True); dpg.add_same_line(); dpg.add_text("", tag="mae_val")

    dpg.add_text("Logs:")
    dpg.add_input_text(tag="log_window", multiline=True, width=500, height=200, readonly=True)

dpg.create_viewport(title="Sea-Ice Prediction MLP")
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Sea-Ice Prediction MLP", True)
dpg.start_dearpygui()
dpg.destroy_context()