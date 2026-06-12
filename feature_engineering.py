import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")

# known slopes for each complexity
COMPLEXITY_SLOPES = {
    "O(1)"      : 0.0,
    "O(log n)"  : 0.3,
    "O(n)"      : 1.0,
    "O(n log n)": 1.1,
    "O(n²)"     : 2.0,
}

print("\n--Complexity Prediction--\n")

predictions = {}

for algo_name in df["algorithm"].unique():
    algo  = df[df["algorithm"] == algo_name]
    X_raw = algo["N"].values
    y     = algo["avg_time"].values

    log_n    = np.log2(X_raw.astype(float))
    log_time = np.log2(y.astype(float))

    # fit a straight line in log-log space → get slope
    slope, intercept = np.polyfit(log_n, log_time, 1)

     # R² on the log-log fit
    log_time_pred = slope * log_n + intercept
    ss_res = np.sum((log_time - log_time_pred) ** 2)
    ss_tot = np.sum((log_time - np.mean(log_time)) ** 2)
    r2 = 1 - ss_res / ss_tot
    confidence = "HIGH" if r2 >= 0.85 else "LOW"
    

    # match slope to nearest known complexity
    predicted = min(COMPLEXITY_SLOPES, 
                    key=lambda k: abs(COMPLEXITY_SLOPES[k] - slope))
    
    predictions[algo_name] = predicted
    
    print(f"{algo_name:<20} → {predicted:<15} slope: {slope:.3f}   R²: {r2:.4f}   {confidence}")



def get_feature(n_array, complexity_label):
    """Extract the single winning feature based on predicted complexity."""
    if complexity_label == "O(1)":
        return np.ones(len(n_array))
    elif complexity_label == "O(log n)":
        return np.log2(n_array + 1e-9)
    elif complexity_label == "O(n)":
        return n_array
    elif complexity_label == "O(n log n)":
        return n_array * np.log2(n_array + 1e-9)
    elif complexity_label == "O(n²)":
        return n_array ** 2
    else:
        return n_array
 
def format_time(seconds, algo_name):
    if algo_name == "fibonacci":
        return "Instantly (Capped Workload)"
    elif seconds < 0.001:
        return f"{seconds * 1000:.4f} ms"
    elif seconds < 60:
        return f"{seconds:.4f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.2f} min"
    elif seconds < 86400:
        return f"{seconds / 3600:.2f} hrs"
    else:
        return f"{seconds / 86400:.2f} days"
 
def power_law(n, c, p):
    return c * (n ** p)
 
TARGET_N = 1_000_000
 
print(f"\n{'=' * 70}")
print(f"  COMPONENT 2 — Runtime Prediction at N={TARGET_N:,}")
print(f"{'=' * 70}")
print(f"\n{'Algorithm':<20} {'LinearReg':>15}   {'CurveFit':>15}   {'p value':>8}")
print("-" * 70)
 
for algo_name in df["algorithm"].unique():
    algo      = df[df["algorithm"] == algo_name]
    X_raw     = algo["N"].values.astype(float)
    y         = algo["avg_time"].values
    predicted = predictions[algo_name]
 
    # --- Linear Regression ---
    X_train = get_feature(X_raw, predicted).reshape(-1, 1)
    lr      = LinearRegression()
    lr.fit(X_train, y)
    X_target   = get_feature(np.array([TARGET_N], dtype=float), predicted).reshape(-1, 1)
    lr_pred    = lr.predict(X_target)[0]
    lr_str     = format_time(lr_pred, algo_name)
 
    # --- Curve Fitting ---
    try:
        params, _ = curve_fit(power_law, X_raw, y, maxfev=5000)
        c, p      = params
        cf_pred   = power_law(TARGET_N, c, p)
        cf_str    = format_time(cf_pred, algo_name)
        p_str     = f"{p:.3f}"
    except:
        cf_str = "fit failed"
        p_str  = "N/A"
 
    print(f"{algo_name:<20} {lr_str:>15}   {cf_str:>15}   {p_str:>8}")
 
print()