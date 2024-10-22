import numpy as np

def eliminate( A, row, pivot_row, alpha):
    A[row] = A[row] + alpha * A[pivot_row]

def scale(A, row, alpha):
    new_r = np.multiply(A[row,], alpha)

def interchange(A, row_i, row_j):
    temp = A[row_i].copy()
    A[row_i], A[row_j] = A[row_j], temp
    
def find_pivot(A, row, col):
    A = np.matrix(A)
    rows_nums = A.shape[0]
    
    for i in range(row, rows_nums):
        if A[i, col] != 0:
            return i
    return -1

def ref(A):
    A = A.astype(float)
    rows,cols = A.shape
    row_now = 0
    
    for col in range(cols): #looks for pivot 
        pivot_row = find_pivot(A, row_now, col)

        if pivot_row == -1:
            continue 

        if pivot_row != row_now: # Swaps current row with the row with pivot
            interchange(A, row_now, pivot_row)
            #changed to match variables was interchange(A, row_i, row_j) before

        scale(A, row_now, 1/A[row_now,col])
        

        for row in range(row_now + 1, rows): #remove rows below pivot
            alpha = -A[row, col] / A[row_now, col]  
            eliminate(A, row, row_now, alpha)

        row_now+=1

        if row_now >=rows:
            break
    return A

A = np.matrix([[-1, 4, -2, -6, -2, 5], [-2, 6, -5, -12, -2, 9],
               [0, -4, -1, 2, 3, -3],[ 0, -4, -1, 2, 3, -3],
               [ 0, 2, 1, 0, -2, 1]])
ref(A)
ref_A = ref(A)

print(ref_A)
