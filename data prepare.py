import sys
import numpy as np
import h5py
import heapq
filename = 'dm3.kc167.example.h5'
f = h5py.File(filename, 'r')
x_train = np.array(f['x_train'])
x_test = np.array(f['x_test'])
x_val = np.array(f['x_val'])
y_train = np.array(f['y_train'])
y_test = np.array(f['y_test'])
y_val= np.array(f['y_val'])
#np.set_printoptions(threshold=sys.maxsize)

#one-hot reverse encoding
x = [[0] * 1000] * 30127
#print(x_val[0])
for i in range(28127):
    a = x_train[i,:,:]
    x[i] = np.argmax(a, axis=1)
for i in range(28127,29127):
    a = x_val[i-28127,:,:]
    x[i] = np.argmax(a, axis=1)
for i in range(29127,30127):
    a = x_test[i-29127,:,:]
    x[i] = np.argmax(a, axis=1)

#k-mers
def k_mer_2D(k,lenth):
    B = [[0] * (4 ** k)] * lenth
    for i in range(lenth):
        A = k_mer_1seq(k, x[i])
        B[i] = A
    return B
def k_mer_1seq(k,input_data):
    A = np.zeros(4 ** k)
    l = len(input_data)
    for i in range(l-k+1):
        loc = 0
        for j in range(k):
            loc = input_data[i+j] * (4**(k-j-1)) + loc
        A[loc] = A[loc] + 1
    return A

def max_len(number, k):
    row_sum = np.zeros(4**k)
    for i in range(4**k):
        row_sum[i] = data[:,i].sum()
    s = heapq.nlargest(number, range(len(row_sum)), row_sum.take)
    index = np.sort(s)
    #np.savetxt('index_k=6.txt', index)
    data_array = data[:,index]
    np.savetxt('k-mer_array_k=6.txt', data_array)

lenth=len(x)
print(lenth)
data = k_mer_2D(6,lenth)
data = np.array(data)
print(data.sum())
max_len(600,6)





