# Algo Complexity Analyzer

An ML pipeline that empirically profiles algorithm runtimes, predicts Big-O complexity using log-log regression, and forecasts execution time at infeasible input sizes using curve fitting and linear regression.


## Demo

Paste your function in `user_function.py` and run:

```bash
python analyzer.py
```

**Example — detecting O(n²) in a duplicate-check function:**

![Demo Output](plots/demo_output.jpg)

**Visualization — ML fit vs actual runtime:**

![ML Fit](plots/regression_curves.png)

![Confusion Matrix](plots/confusion_matrix.png)


## How It Works

### Step 1 — Empirical Profiling
Each function is run across multiple input sizes N = [100, 500, 1000, 2000, 5000, 10000].
Timing is averaged over 7 trials to reduce noise. Fast functions (< 0.1ms) 
automatically use `timeit` with 2000 repetitions for stable nanosecond measurements.
Slow functions use `time.perf_counter` with fresh unique input arrays each trial 
to guarantee worst-case behavior.

### Step 2 The Math — Log-Log Regression (Complexity Classification)

An algorithm's runtime follows a fundamental power-law relationship:

$$\text{Time} = c \cdot N^p$$

Where:
- $N$ = input size
- $p$ = the complexity exponent (e.g. $2.0$ for a quadratic algorithm)
- $c$ = hardware constant (governed by CPU frequency and memory architecture)

---

Fitting a curved power-law directly introduces errors across varying scales.
To linearize this, we take the base-2 logarithm of both sides:

$$\log_2(\text{Time}) = \log_2(c \cdot N^p)$$

Applying logarithm laws $(\log(A \cdot B) = \log A + \log B$ and $\log(A^p) = p \cdot \log A)$:

$$\log_2(\text{Time}) = p \cdot \log_2(N) + \log_2(c)$$

This maps directly to the classic straight-line formula:

$$y = m \cdot x + b$$

Where:
- $y = \log_2(\text{Time})$
- $x = \log_2(N)$
- $m = p$ → **the slope IS the Big-O exponent**
- $b = \log_2(c)$ → hardware constant (intercept)

---

We run a 1st-degree Ordinary Least Squares polynomial fit (`np.polyfit`) 
across these transformed coordinates to extract slope $m$.

The slope is then matched to the nearest known complexity class:

| Slope $p$ | Complexity Class |
|:---:|:---:|
| $p \approx 0.0$ | $O(1)$ |
| $p \approx 0.3$ | $O(\log n)$ |
| $p \approx 1.0$ | $O(n)$ |
| $p \approx 1.1$ | $O(n \log n)$ |
| $p \approx 2.0$ | $O(n^2)$ |

R² is computed on the log-log fit to measure confidence of the straight-line assumption:

$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- $R^2 \geq 0.85$ → **HIGH confidence**
- $R^2 < 0.85$ → **LOW confidence** (noisy data or capped workload)

### Step 3 — Runtime Prediction at N = 1,000,000
Two independent methods predict runtime at scales never actually profiled:
#### Isolated Linear Regression (`scikit-learn`)
To prevent multi-collinearity and parameter drift when projecting values outward by multiple orders of magnitude, the predictor isolates the single winning mathematical feature:

* If $O(n^2)$ is chosen, the dataset is restricted to a single feature column containing values scaled exclusively to $N^2$.
* A `LinearRegression()` model fits this isolated matrix directly against the raw seconds vector ($y$). This directly extracts the exact hardware execution coefficient ($c$) without scaling distortion, providing a clean projection vector to evaluate the target input workload.

#### Direct Power-Law Curve Fitting (`scipy.optimize`)
As an independent structural validation mechanism, the script applies a non-linear least-squares optimization using the Levenberg-Marquardt algorithm via `curve_fit`.


Agreement between both methods validates the prediction. The exponent `p` 
from curve fitting independently confirms the complexity class detected in Step 2.

### Note on Exponential Algorithms (O(2ⁿ))
Truly exponential algorithms like naive Fibonacci are **infeasible to 
profile empirically** — `fib(50)` would take years to compute.

We cap the workload at `fib(25)` regardless of input size N. As a result, 
the profiler correctly detects near-constant runtime and classifies it as 
O(1) — which is honest: the *profiled workload* is constant, even though 
the *theoretical complexity* is O(2ⁿ).

This is a deliberate design decision, not a bug. In practice, exponential 
algorithms are identified via theoretical analysis, not empirical profiling.


## Results

### Complexity Classification

| Algorithm | True Complexity | Predicted | Slope | R² | Confidence |
|-----------|----------------|-----------|-------|----|------------|
| linear_search | O(n) | O(n) | ~1.0 | 0.99 | HIGH |
| bubble_sort | O(n²) | O(n²) | ~2.0 | 0.99 | HIGH |
| binary_search | O(log n) | O(log n) | ~0.26 | 0.93 | HIGH |
| merge_sort | O(n log n) | O(n log n) | ~1.1 | 0.99 | HIGH |
| fibonacci | O(2ⁿ) | O(1) [Capped] | ~0.0 | LOW | — |

### Runtime Prediction at N = 1,000,000

| Algorithm | Linear Regression | Curve Fitting | p value |
|-----------|------------------|---------------|---------|
| linear_search | 0.024 sec | 0.022 sec | 0.983 |
| bubble_sort | 9.86 hrs | 11.14 hrs | 2.031 |
| binary_search | 0.013 ms | 0.019 ms | 0.199 |
| merge_sort | 2.47 sec | 2.12 sec | 1.047 |
| fibonacci | Instantly (Capped) | Instantly (Capped) | — |

---

## Project Structure

The repository is split into two parts: an offline script to gather baseline data on standard algorithms, and a live sandbox tool to test any custom function.

```text
algo-complexity-analyzer/
│
├── profiler.py             # Benchmarks the 5 built-in algorithms and saves timings to CSV
├── feature_engineering.py  # Runs the log-log classification and runtime predictions on the baseline data
├── analyzer.py             # The main CLI tool that profiles and predicts complexity for any custom function
├── user_function.py        # Scratchpad file where you paste your own code to be tested
├── data.csv                # The raw execution timing dataset generated by profiler.py
└── README.md               # Math breakdowns, logic explanations, and results

          
