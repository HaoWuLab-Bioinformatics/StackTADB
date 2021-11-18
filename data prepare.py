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

#one-hot reverse encoding,Sequence in numbers
x = [[0] * 1000] * 30127

for i in range(28127):
    a = x_train[i,:,:]
    x[i] = np.argmax(a, axis=1)
for i in range(28127,29127):
    a = x_val[i-28127,:,:]
    x[i] = np.argmax(a, axis=1)
for i in range(29127,30127):
    a = x_test[i-29127,:,:]
    x[i] = np.argmax(a, axis=1)

x = np.array(x)
#np.savetxt('data.txt',x)

np.savetxt('data.txt',x)
#Sequence in bases
data = np.loadtxt('data.txt')
outf = "data_base.txt"
out=open(outf,'w')
for i in range(len(data)):
    for j in range(len(data[0])):
        if data[i][j] == 0 :#A
            out.writelines('A')
        elif data[i][j] == 1 :#T
            out.writelines('T')
        elif data[i][j] == 2 :#G
            out.writelines('G')
        else :#C
            out.writelines('C')
    out.writelines('\n')
out.close()

#Convert .txt files to .fa files
inf = "data_base.txt"
outf= "data_base.fasta"
def readwrite(inf,outf):
    f=open(inf,'r')
    out=open(outf,'w')
    i=1
    for line in f.readlines():
        list_line = line.strip().split()
        x=list_line[0]+"\n" 
        y=">Chr"+str(i)+"\n"  
        out.writelines(y)
        out.writelines(x)
        i=i+1
    f.close()
    out.close()
readwrite(inf,outf)






