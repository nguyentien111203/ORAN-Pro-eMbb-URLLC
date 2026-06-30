import os
import numpy as np
import matplotlib.pyplot as plt


def moving_average(x, window=100):
    x = np.asarray(x)

    if len(x) < window:
        return x

    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def plot_rate(rate_dict,
              service_type="URLLC",
              rate_type="avg",
              save_dir="results",
              window=100):

    os.makedirs(save_dir, exist_ok=True)

    service_type = service_type.upper()

    if service_type == "URLLC":
        min_rate = np.asarray(rate_dict["min_urllc"])
        avg_rate = np.asarray(rate_dict["avg_urllc"])
    elif service_type == "EMBB":
        min_rate = np.asarray(rate_dict["min_embb"])
        avg_rate = np.asarray(rate_dict["avg_embb"])
    else:
        raise ValueError("service_type must be URLLC or eMBB")

    if rate_type.lower() == "min":
        y = min_rate
        title = f"{service_type} Min Rate"

    elif rate_type.lower() == "avg":
        y = avg_rate
        title = f"{service_type} Average Rate"

    elif rate_type.lower() == "gap":
        y = avg_rate - min_rate
        title = f"{service_type} Fairness Gap"

    else:
        raise ValueError("rate_type must be min, avg, or gap")

    y = moving_average(y, window)

    plt.figure(figsize=(10, 5))

    plt.plot(y)

    if rate_type.lower() != "gap":
        plt.axhline(
            y=1.0,
            linestyle="--",
            linewidth=1.5,
            label="QoS Threshold"
        )
        plt.legend()

    plt.xlabel("Episode")
    plt.ylabel("Rate")
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()

    filename = os.path.join(
        save_dir,
        f"{service_type}_{rate_type}_SAC.png"
    )

    plt.savefig(filename, dpi=300)
    plt.close()

    print(f"Saved: {filename} SAC")