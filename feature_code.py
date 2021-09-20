import numpy as np
import math
import heapq
from itertools import combinations, combinations_with_replacement, permutations
#from repDNA.psenac import PseDNC
data = np.loadtxt('data.txt')
print(data.shape)

#normalization
def noramlization(data):
    minVals = data.min(0)
    maxVals = data.max(0)
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData

#Keep the top number features with the largest count
def max_len(number, k, data):
    row_sum = np.zeros(4**k)
    for i in range(4**k):
        row_sum[i] = data[:,i].sum()
    s = heapq.nlargest(number, range(len(row_sum)), row_sum.take)
    index = np.sort(s)
    #np.savetxt('index_k=6.txt', index)
    data_array = data[:,index]
    return data_array

#########################feature k-mers code#############################
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


###########################feature NPSE code##############################
def NPSE(k):
    number = len(data)
    length = len(data[0])
    feature_NPSE = np.array([[0]*(4*4*(k+1))]*number)
    print(feature_NPSE.shape)
    for i in range(number):
        for s in range(k+1):
            for j in range(length-(s+1)):
                pos = int(data[i][j]*4 +data[i][j+(s+1)])
                feature_NPSE[i][pos+(4*4*s)] = feature_NPSE[i][pos+(4*4*s)] + 1
    return feature_NPSE

###########################feature PSSM code##############################
def PSSM():
    number = len(data)
    length = len(data[0])
    PFM = np.array([[0] * length] * 4)
    #PPM = np.array([[0] * length] * 4)
    #PSSM = np.array([[0] * length] * 4)
    for i in range(number):
        for j in range(length):
            pos = int(data[i][j])
            PFM[pos][j] = PFM[pos][j] + 1
    print(PFM)
    PPM = PFM / number
    PSSM = np.log2(PPM / 0.25)
    print(PSSM)
    feature_PSSM = np.array([[0.0] * length] * number)
    print(feature_PSSM.shape)
    for i in range(number):
        for j in range(length):
            pos = int(data[i][j])
            feature_PSSM[i,j] = PSSM[pos][j]
    return feature_PSSM

###########################feature Mismatch code##############################
def Mismatch_k_tuple(k):
    number = len(data)
    length = len(data[0])
    base = np.array([0,1,2,3])
    feature_Mismatch = np.array([[0] * (4 ** k)] * number)
    print(feature_Mismatch.shape)
    for i in range(number):
        for j in range(length - k + 1):
            subsequence = data[i][j:j + k]
            #print(subsequence)
            loc = 0
            for l in range(k):
                loc = int(subsequence[l]*(4**(k-l-1))) + loc
            feature_Mismatch[i][loc] = feature_Mismatch[i][loc] + 1
            for l in range(k):
                substitution = subsequence
                '''print(base)
                index = int(subsequence[l])
                base = np.delete(base,index)
                print(base)'''
                for letter in base:
                    #print(letter)
                    #print(subsequence[l])
                    if letter != subsequence[l]:
                        substitution = list(substitution)
                        substitution[l] = letter
                        #print('substitution',substitution)
                        pos = 0
                        for m in range(k):
                            pos = int(substitution[m] * (4 ** (k - m - 1))) + pos
                        feature_Mismatch[i][pos] = feature_Mismatch[i][pos] + 1
    return feature_Mismatch


###########################feature subsequence_profile code##############################

def subsequence_profile(k,delta):
    number = len(data)
    length = len(data[0])
    feature_subsequence_profile = np.array([[0.0] * (4 ** k)] * number)
    for i in range(number):
        print(i)
        for j in range(length-k+1):
            if j < length-10:
                index_lst = list(combinations(range(j, j + 10), k))
                index_k_sub_1 = len(list(combinations(range(j + 1, j + 10), k - 1)))#combinations k-1
                index_lst = index_lst[0:index_k_sub_1]
            else:
                index_lst = list(combinations(range(j, length), k))
                index_k_sub_1 = len(list(combinations(range(j + 1, length), k - 1)))  # combinations k-1
                index_lst = index_lst[0:index_k_sub_1]
            #print(index_lst)
            for subseq_index in index_lst:
                subseq_index = list(subseq_index)
                subsequence = data[i][subseq_index]
                loc = 0
                for l in range(k):
                    loc = int(subsequence[l]*(4**(k-l-1))) + loc
                subseq_length = subseq_index[-1] - subseq_index[0] + 1
                subseq_score = 1 if subseq_length == k else delta ** subseq_length
                feature_subsequence_profile[i][loc] = feature_subsequence_profile[i][loc] + subseq_score
                #print(feature_subsequence_profile[i])
    return feature_subsequence_profile

###########################feature Natural vector code##############################

def NV(k):
    number = len(data)
    length = len(data[0])
    feature_NV = np.array([[0.0] * (4 ** k * 3)] * number)
    for i in range(number):
        print(i)
        nk_frequence = [0] * (4 ** k)  # 4
        tk_position = [0] * (4 ** k)  # 4
        u = [0] * (4 ** k)  # 4
        for j in range(0, length - k + 1):
            subsequence = data[i][j:j + k]
            loc = 0
            for l in range(k):
                loc = int(subsequence[l] * (4 ** (k - l - 1))) + loc
            nk_frequence[loc] += 1
            tk_position[loc] += j + 1
        for j in range(0, 4 ** k):
            if nk_frequence[j] == 0:
                u[j] = 0
            else:
                u[j] = tk_position[j] / nk_frequence[j]
        dk_secondery = [0] * (4 ** k)
        for j in range(0, length - k + 1):
            subsequence = data[i][j:j + k]
            loc = 0
            for l in range(k):
                loc = int(subsequence[l] * (4 ** (k - l - 1))) + loc
            dk_secondery[loc] += ((j + 1 - u[loc]) ** 2) / (nk_frequence[loc] * length)
        for j in range(0, 4 ** k):
            feature_NV[i][j * 3] = nk_frequence[j]
            feature_NV[i][j * 3 + 1] = u[j]
            feature_NV[i][j * 3 + 2] = dk_secondery[j]
    return feature_NV
#k-mers
'''k=5
x = data
lenth=len(x)
data = k_mer_2D(6,lenth)
data = np.array(data)
feature1 = max_len(600,6,data)
#feature1 = noramlization(feature1)
np.savetxt('feature_k-mers_k='+str(k)+'.txt', data_array)'''

#NPSE
k=7
feature2 = NPSE(k)
np.savetxt('feature_NPSE_k='+str(k)+'.txt',feature2)

#PSSM
'''feature3 = PSSM()
np.savetxt('feature_PSSM.txt',feature3)'''

#Mismatch_k_tuple
'''k=8
feature_4 = Mismatch_k_tuple(k)
feature4 = max_len(600,k,feature_4)
print(feature4.shape)
np.savetxt('feature_Mismatch_k='+str(k)+'.txt',feature4)'''

#subsequence_profile
'''k=1
delta=0.5
feature_5 = subsequence_profile(k,delta)
feature5 = max_len(600,k,feature_5)
np.savetxt('feature_subsequence_profile_k='+str(k)+'_delta='+str(delta)+'.txt',feature5)'''

#Natural vector
'''k=1
feature6 = NV(k)
np.savetxt('feature_NV_k='+str(k)+'.txt',feature6)'''
