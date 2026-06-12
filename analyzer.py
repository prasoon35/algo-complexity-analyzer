import time
import timeit
import random
import numpy as np
from user_function import mystery

# ---- SETUP -------------------------------------------------------------------------
INPUT_SIZES_SLOW = [100, 500, 1000, 2000, 5000, 10000]
INPUT_SIZES_FAST = [100, 500, 1000, 2000, 5000, 10000, 75000]
N_TRIALS    = 7

dataset = []

print("Profiling your function...\n")

# auto detect if function is fast or slow
# run once first to check
test_arr = [random.randint(0, 10000) for _ in range(1000)]
start = time.perf_counter()
mystery(test_arr)
test_time = time.perf_counter() - start

USE_TIMEIT = test_time < 0.001  # faster than 1ms → use timeit
print(f"Function speed: {'FAST — using timeit' if USE_TIMEIT else 'SLOW — using perf_counter'}\n")


INPUT_SIZES = INPUT_SIZES_FAST if USE_TIMEIT else INPUT_SIZES_SLOW

for n in INPUT_SIZES:
    arr = [random.randint(0, 10000) for _ in range(n)]
    if USE_TIMEIT:
        t   = timeit.timeit(lambda: mystery(arr), number=2000)
        avg = t / 200
    else:
        times = []
        for _ in range(N_TRIALS):
            arr = random.sample(range(n * 10), n)
            start = time.perf_counter()
            mystery(arr)
            times.append(time.perf_counter() - start)
        avg = sum(times) / N_TRIALS
    
    dataset.append((n, avg))
    print(f"N={n:<6} → {avg:.8f} sec")

print("\nProfiling complete.")





# ---- CLASSIFICATION -------------------------------------------------------
X_raw = np.array([n for n, t in dataset])
y     = np.array([t for n, t in dataset])

log_n    = np.log2(X_raw.astype(float))
log_time = np.log2(y.astype(float))

slope, intercept = np.polyfit(log_n, log_time, 1)

# R²
log_time_pred = slope * log_n + intercept
ss_res = np.sum((log_time - log_time_pred) ** 2)
ss_tot = np.sum((log_time - np.mean(log_time)) ** 2)
r2     = 1 - ss_res / ss_tot
confidence = "HIGH" if r2 >= 0.85 else "LOW"

COMPLEXITY_SLOPES = {
    "O(1)"      : 0.0,
    "O(log n)"  : 0.3,
    "O(n)"      : 1.0,
    "O(n log n)": 1.1,
    "O(n²)"     : 2.0,
}

predicted = min(COMPLEXITY_SLOPES,
                key=lambda k: abs(COMPLEXITY_SLOPES[k] - slope))

print(f"\n{'='*45}")
print(f"  Detected complexity : {predicted}")
print(f"  Slope               : {slope:.3f}")
print(f"  R²                  : {r2:.4f}")
print(f"  Confidence          : {confidence}")
print(f"{'='*45}")






# ---- RUNTIME PREDICTION -----------------------------------------------------------
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

TARGET_N = 1_000_000

def get_feature(n_array, label):
    if label == "O(1)":
        return np.ones(len(n_array))
    elif label == "O(log n)":
        return np.log2(n_array + 1e-9)
    elif label == "O(n)":
        return n_array
    elif label == "O(n log n)":
        return n_array * np.log2(n_array + 1e-9)
    elif label == "O(n²)":
        return n_array ** 2

def format_time(sec):
    if sec < 0.001:   return f"{sec*1000:.4f} ms"
    elif sec < 60:    return f"{sec:.4f} sec"
    elif sec < 3600:  return f"{sec/60:.2f} min"
    elif sec < 86400: return f"{sec/3600:.2f} hrs"
    else:             return f"{sec/86400:.2f} days"

# linear regression
X_train = get_feature(X_raw.astype(float), predicted).reshape(-1, 1)
lr = LinearRegression()
lr.fit(X_train, y)
X_target = get_feature(np.array([TARGET_N], dtype=float), predicted).reshape(-1, 1)
lr_pred  = lr.predict(X_target)[0]

# curve fitting
def power_law(n, c, p):
    return c * (n ** p)

params, _ = curve_fit(power_law, X_raw.astype(float), y, maxfev=5000)
c, p = params
cf_pred = power_law(TARGET_N, c, p)

print(f"\n  Predicted at N=1,000,000")
print(f"  Linear Regression : {format_time(lr_pred)}")
print(f"  Curve Fitting     : {format_time(cf_pred)}  (p={p:.3f})")
print(f"{'='*45}\n")