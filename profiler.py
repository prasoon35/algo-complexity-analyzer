import time
import random
import csv

# --- ALGORITHM 1: Linear Search O(N) ---
def linear_search(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return True
    return False

# --- ALGORITHM 2: Bubble Sort O(N^2) ---
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

# --- ALGORITHM 3: Binary Search O(log N) ---
def binary_search(arr, target):
    low, high = 0, len(arr) - 1
    while low <= high:
        mid = low + (high - low) // 2
        if arr[mid] == target:
            return True
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return False

# --- ALGORITHM 4: Merge Sort O(N log N) ---
def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        L, R = arr[:mid], arr[mid:]
        merge_sort(L)
        merge_sort(R)
        i = j = k = 0
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]; i += 1
            else:
                arr[k] = R[j]; j += 1
            k += 1
        while i < len(L):
            arr[k] = L[i]; i += 1; k += 1
        while j < len(R):
            arr[k] = R[j]; j += 1; k += 1

# --- ALGORITHM 5: Fibonacci O(2^N) ---
def naive_fib(arr):
    n = min(len(arr), 25)
    def fib(n):
        if n <= 1: return n
        return fib(n-1) + fib(n-2)
    fib(n)


# --- SETUP ------------------------------------------------------
INPUT_SIZES = [100, 500, 1000, 2000, 5000, 10000, 20000]
N_TRIALS    = 7

linear_dataset   = []
bubble_dataset   = []
binary_dataset   = []
merge_dataset    = []
fibonacci_dataset = []

print("Starting Data Collection...\n")

for n in INPUT_SIZES:
    linear_times   = []
    bubble_times   = []
    binary_times   = []
    merge_times    = []
    fibonacci_times = []

    for _ in range(N_TRIALS):
        arr = [random.randint(0, 10000) for _ in range(n)]
        target = -1  # worst case — not in array

        # linear search
        start = time.perf_counter()
        linear_search(arr, target)
        linear_times.append(time.perf_counter() - start)

        # bubble sort
        arr_copy = arr.copy()
        start = time.perf_counter()
        bubble_sort(arr_copy)
        bubble_times.append(time.perf_counter() - start)

        # binary search — reuse sorted copy from bubble sort
        start = time.perf_counter()
        binary_search(arr_copy, target)
        binary_times.append(time.perf_counter() - start)

        # merge sort
        arr_copy = arr.copy()
        start = time.perf_counter()
        merge_sort(arr_copy)
        merge_times.append(time.perf_counter() - start)

        # fibonacci
        start = time.perf_counter()
        naive_fib(arr)
        fibonacci_times.append(time.perf_counter() - start)

    
    avg_linear    = sum(linear_times)    / N_TRIALS
    avg_bubble    = sum(bubble_times)    / N_TRIALS
    avg_binary    = sum(binary_times)    / N_TRIALS
    avg_merge     = sum(merge_times)     / N_TRIALS
    avg_fibonacci = sum(fibonacci_times) / N_TRIALS


    linear_dataset.append((n, avg_linear))
    bubble_dataset.append((n, avg_bubble))
    binary_dataset.append((n, avg_binary))
    merge_dataset.append((n, avg_merge))
    fibonacci_dataset.append((n, avg_fibonacci))

    print(f"N={n:<6} | Linear: {avg_linear:.6f} | Bubble: {avg_bubble:.6f} | Binary: {avg_binary:.6f} | Merge: {avg_merge:.6f} | Fib: {avg_fibonacci:.6f}")

print("\nCollection Complete!")

# --- SAVE TO CSV ---
with open("data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["algorithm", "N", "avg_time"])
    for n, t in linear_dataset:
        writer.writerow(["linear_search", n, t])
    for n, t in bubble_dataset:
        writer.writerow(["bubble_sort", n, t])
    for n, t in binary_dataset:
        writer.writerow(["binary_search", n, t])
    for n, t in merge_dataset:
        writer.writerow(["merge_sort", n, t])
    for n, t in fibonacci_dataset:
        writer.writerow(["fibonacci", n, t])

print("Saved to data.csv")