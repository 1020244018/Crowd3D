import numpy as np

def distance(p1,p2):
    return np.sqrt(np.sum(np.square(p1-p2)))
# FPS sampling
def FPS(samples, sampling_num):
    N=samples.shape[0]
    center=np.mean(samples,axis=0)
    select_indexes, L=[], []
    for i in range(N):
        L.append(distance(samples[i],center))
    p0_i=np.argmax(L)
    L=[]
    for i in range(N):
        L.append(distance(samples[i],samples[p0_i]))
    p1_i=np.argmax(L)
    select_indexes+=[p0_i,p1_i]
    for j in range(sampling_num-2):
        for i in range(N):
            d=distance(samples[i],samples[select_indexes[-1]])
            if d<=L[i]:
                L[i]=d
        select_indexes.append(np.argmax(L))
    return select_indexes