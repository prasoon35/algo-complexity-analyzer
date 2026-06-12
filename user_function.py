# -----PASTE YOUR FUNCTION HERE ----------------------
# Rules:
#   1. Function must be named  mystery(arr)
#   2. arr is a Python list of random integers of length N
#   3. Do not change the function name

def mystery(arr):
    # example — replace with your own code
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] == arr[j]:
                return True
    return False