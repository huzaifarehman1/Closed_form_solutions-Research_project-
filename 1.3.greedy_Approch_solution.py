import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn

def solve_mse(X, Y):
    """
    Solve the minimum-MSE linear regression problem:

        min_W ||XW - Y||_F^2

    using the Moore-Penrose pseudoinverse.

    Parameters
    ----------
    X : np.ndarray
        Input/design matrix of shape (n_samples, n_features).

    Y : np.ndarray
        Target matrix of shape (n_samples, n_outputs).

    Returns
    -------
    W : np.ndarray
        Closed-form minimum-MSE solution.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)

    return np.linalg.pinv(X) @ Y


# for task 1 lets see how our model performs to learn x^2 




# ============================================================
# 1. Create a simple nonlinear regression dataset
# ============================================================

np.random.seed(42)
torch.manual_seed(42)

n = 500

X = np.random.uniform(-5, 5, (n, 1))

# Nonlinear target
Y = 3 * X**2 + 2 * X + 5 + np.random.normal(0, 3, (n, 1))


# ============================================================
# 2. Train / test split
# ============================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 3. Standardization
# ============================================================

X_scaler = StandardScaler()
Y_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

Y_train = Y_scaler.fit_transform(Y_train)
Y_test = Y_scaler.transform(Y_test)


# ============================================================
# 4. Convert to PyTorch tensors
# ============================================================

X_train_t = torch.tensor(X_train, dtype=torch.float32)
Y_train_t = torch.tensor(Y_train, dtype=torch.float32)

X_test_t = torch.tensor(X_test, dtype=torch.float32)
Y_test_t = torch.tensor(Y_test, dtype=torch.float32)


# ============================================================
# 5. Standard neural network
# ============================================================

model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),

    nn.Linear(32, 32),
    nn.Tanh(),

    nn.Linear(32, 1)
)


# ============================================================
# 6. Loss + optimizer
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01
)


# ============================================================
# 7. Train
# ============================================================

epochs = 200

loss_history = []

for epoch in range(epochs):

    optimizer.zero_grad()

    prediction = model(X_train_t)

    loss = criterion(prediction, Y_train_t)

    loss.backward()

    optimizer.step()

    loss_history.append(loss.item())

    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch {epoch + 1:4d} | "
            f"Train MSE: {loss.item():.6f}"
        )


# ============================================================
# 8. Evaluate
# ============================================================

model.eval()

with torch.no_grad():

    train_pred = model(X_train_t).numpy()
    test_pred = model(X_test_t).numpy()


train_mse = mean_squared_error(Y_train, train_pred)
test_mse = mean_squared_error(Y_test, test_pred)

print("\n==============================")
print("RESULTS")
print("==============================")

print(f"Standardized Train MSE: {train_mse:.6f}")
print(f"Standardized Test MSE:  {test_mse:.6f}")


# ============================================================
# 9. Convert predictions back to original scale
# ============================================================

train_pred_original = Y_scaler.inverse_transform(train_pred)
test_pred_original = Y_scaler.inverse_transform(test_pred)

Y_train_original = Y_scaler.inverse_transform(Y_train)
Y_test_original = Y_scaler.inverse_transform(Y_test)

original_test_mse = mean_squared_error(
    Y_test_original,
    test_pred_original
)

print(f"Original-scale Test MSE: {original_test_mse:.6f}")





# lets try greedy approch

def greedy_nn(X, Y, n=3, activation=np.tanh, test_size=0.2,
              random_state=42, plot=True):
    """
    Greedy layer-wise neural network using closed-form
    minimum-MSE regression at every layer.

    Parameters
    ----------
    X : np.ndarray
        Input data, shape (n_samples, n_features)

    Y : np.ndarray
        Target data, shape (n_samples, n_outputs)

    n : int
        Number of network layers.

    activation : function
        Activation function. Default: np.tanh

    test_size : float
        Fraction of data used for testing.

    random_state : int
        Random seed for train/test split.

    plot : bool
        Whether to generate plots.

    Returns
    -------
    weights : list
        List containing W1, W2, ..., Wn.

    model : dict
        Contains scalers, predictions, errors, and intermediate
        representations.
    """

    # --------------------------------------------------------
    # Make sure X and Y are 2D
    # --------------------------------------------------------

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)


    # --------------------------------------------------------
    # Train / test split
    # --------------------------------------------------------

    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        random_state=random_state
    )


    # --------------------------------------------------------
    # Standardization
    # --------------------------------------------------------

    X_scaler = StandardScaler()
    Y_scaler = StandardScaler()

    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)

    Y_train = Y_scaler.fit_transform(Y_train)
    Y_test = Y_scaler.transform(Y_test)


    # --------------------------------------------------------
    # Greedy training
    # --------------------------------------------------------

    weights = []

    # Current representation
    V_train = X_train
    V_test = X_test

    # Store intermediate representations
    representations_train = [V_train]
    representations_test = [V_test]

    # Error after each layer
    train_mse_by_layer = []
    test_mse_by_layer = []


    print("\n" + "=" * 60)
    print("GREEDY NEURAL NETWORK")
    print("=" * 60)

    print(f"Layers:       {n}")
    print(f"Activation:   {activation.__name__}")
    print(f"Train size:   {len(X_train)}")
    print(f"Test size:    {len(X_test)}")


    # ========================================================
    # Layer-by-layer solution
    # ========================================================

    for layer in range(1, n + 1):

        # -----------------------------------------------
        # Solve W using closed-form MSE
        # -----------------------------------------------

        W = solve_mse(V_train, Y_train)

        weights.append(W)


        # -----------------------------------------------
        # Linear output of this layer
        # -----------------------------------------------

        Z_train = V_train @ W
        Z_test = V_test @ W


        # -----------------------------------------------
        # Calculate error BEFORE activation
        #
        # This is the exact MSE solved by this regression.
        # -----------------------------------------------

        train_mse = mean_squared_error(
            Y_train,
            Z_train
        )

        test_mse = mean_squared_error(
            Y_test,
            Z_test
        )

        train_mse_by_layer.append(train_mse)
        test_mse_by_layer.append(test_mse)


        print(
            f"\nLayer {layer}"
            f"\n  W shape:       {W.shape}"
            f"\n  Train MSE:     {train_mse:.8f}"
            f"\n  Test MSE:      {test_mse:.8f}"
        )


        # -----------------------------------------------
        # Apply activation ONLY if this is not
        # the final/output layer.
        # -----------------------------------------------

        if layer < n:

            V_train = activation(Z_train)
            V_test = activation(Z_test)

            representations_train.append(V_train)
            representations_test.append(V_test)


    # ========================================================
    # Final prediction
    # ========================================================

    Y_pred_train_scaled = Z_train
    Y_pred_test_scaled = Z_test


    # --------------------------------------------------------
    # Convert predictions back to original Y scale
    # --------------------------------------------------------

    Y_pred_train = Y_scaler.inverse_transform(
        Y_pred_train_scaled
    )

    Y_pred_test = Y_scaler.inverse_transform(
        Y_pred_test_scaled
    )

    Y_train_original = Y_scaler.inverse_transform(Y_train)
    Y_test_original = Y_scaler.inverse_transform(Y_test)


    # --------------------------------------------------------
    # Original-scale MSE
    # --------------------------------------------------------

    original_train_mse = mean_squared_error(
        Y_train_original,
        Y_pred_train
    )

    original_test_mse = mean_squared_error(
        Y_test_original,
        Y_pred_test
    )


    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    print(
        f"Standardized Train MSE: "
        f"{train_mse_by_layer[-1]:.8f}"
    )

    print(
        f"Standardized Test MSE:  "
        f"{test_mse_by_layer[-1]:.8f}"
    )

    print(
        f"Original-scale Train MSE: "
        f"{original_train_mse:.8f}"
    )

    print(
        f"Original-scale Test MSE:  "
        f"{original_test_mse:.8f}"
    )


    # ========================================================
    # Plotting
    # ========================================================

    if plot:

        # ----------------------------------------------------
        # MSE by layer
        # ----------------------------------------------------

        layers = np.arange(1, n + 1)

        plt.figure(figsize=(8, 5))

        plt.plot(
            layers,
            train_mse_by_layer,
            marker="o",
            label="Train MSE"
        )

        plt.plot(
            layers,
            test_mse_by_layer,
            marker="o",
            label="Test MSE"
        )

        plt.xlabel("Layer")
        plt.ylabel("MSE")
        plt.title("Greedy Neural Network: MSE by Layer")

        plt.xticks(layers)
        plt.grid()
        plt.legend()

        plt.show()


        # ----------------------------------------------------
        # Training representations
        # ----------------------------------------------------

        plt.figure(figsize=(8, 5))

        plt.scatter(
            X_train[:, 0],
            Y_train[:, 0],
            label="True Y"
        )

        plt.scatter(
            X_train[:, 0],
            Y_pred_train_scaled[:, 0],
            label="Prediction"
        )

        plt.xlabel("Standardized X")
        plt.ylabel("Standardized Y")

        plt.title("Greedy Neural Network Predictions")

        plt.grid()
        plt.legend()

        plt.show()


        # ----------------------------------------------------
        # Original-scale prediction
        # ----------------------------------------------------

        plt.figure(figsize=(8, 5))

        plt.scatter(
            X_test[:, 0],
            Y_test_original[:, 0],
            label="True"
        )

        plt.scatter(
            X_test[:, 0],
            Y_pred_test[:, 0],
            label="Prediction"
        )

        plt.xlabel("Standardized X")
        plt.ylabel("Y")

        plt.title("Final Predictions on Test Set")

        plt.grid()
        plt.legend()

        plt.show()


        # ----------------------------------------------------
        # Predicted vs actual
        # ----------------------------------------------------

        plt.figure(figsize=(6, 6))

        plt.scatter(
            Y_test_original,
            Y_pred_test
        )

        min_value = min(
            Y_test_original.min(),
            Y_pred_test.min()
        )

        max_value = max(
            Y_test_original.max(),
            Y_pred_test.max()
        )

        plt.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--"
        )

        plt.xlabel("Actual Y")
        plt.ylabel("Predicted Y")

        plt.title("Predicted vs Actual")

        plt.grid()

        plt.show()


    # ========================================================
    # Return everything useful
    # ========================================================

    model = {
        "weights": weights,

        "X_scaler": X_scaler,
        "Y_scaler": Y_scaler,

        "X_train": X_train,
        "X_test": X_test,

        "Y_train": Y_train,
        "Y_test": Y_test,

        "Y_pred_train": Y_pred_train,
        "Y_pred_test": Y_pred_test,

        "representations_train": representations_train,
        "representations_test": representations_test,

        "train_mse_by_layer": train_mse_by_layer,
        "test_mse_by_layer": test_mse_by_layer,

        "train_mse": original_train_mse,
        "test_mse": original_test_mse,
    }

    return weights, model


weights, model = greedy_nn(
    X,
    Y,
    n=70
)



### RESULTS