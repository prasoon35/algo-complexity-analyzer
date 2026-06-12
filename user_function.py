# -----PASTE YOUR FUNCTION HERE ----------------------
# Rules:
#   1. Function must be named  mystery(arr)
#   2. arr is a Python list of random integers of length N
#   3. Do not change the function name

def mystery(arr):
    # example — replace with your own code
    low, high = 0, len(arr) - 1
    while low < high:
        mid = low + (high - low) // 2
        if arr[mid] < arr[mid + 1]:
            low = mid + 1
        else:
            high = mid
    return low