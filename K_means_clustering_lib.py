import numpy as np
import pandas as pd

# k = [2,3,4,5]
# no_examples = 549
# random_indices = np.zeros((100,k),dtype=int)
# cost = np.zeros((100),dtype=int)
# initial_centroid_3d =np.zeros((100,6,k), dtype=int)
def random_indices(no_examples, k):
    random_indices = np.zeros((100,k),dtype=int)
    for i in range (100):        
        random_indices[i] = np.random.choice(np.arange(1,no_examples+1), size=k, replace=False)

def initial__centroid(x,k,random_indices):
    initial_centroid_3d =np.zeros((100,6,k), dtype=int)
    cost = np.zeros((100),dtype=int)
    initial_centroid = x[random_indices,:]
    x_reshaped = x[:, :, np.newaxis]  # Shape becomes (549, 6, 1)
    for i in range(100):
        p = initial_centroid[i]
        print(p)
    #     initial_centroid_3d[i]= p.T.reshape(1, 6, 3)
    #     print(initial_centroid_3d[i])
    #     z = np.sum((initial_centroid_3d[i] - x_reshaped)*(initial_centroid_3d[i] - x_reshaped), axis=1)
    #     cost[i] = np.sum(np.min(z,axis=1))
    
    # return initial_centroid_3d[np.argmin(cost)]

def cluster(centroid,x_reshaped):
    z = np.sum((centroid - x_reshaped)*(centroid - x_reshaped), axis=1)
    centroid_index = np.argmin(z, axis=1)
    result = [(row, col.item()) for row, col in enumerate(centroid_index)]
    print(f"the list represents (training_expl_index, centroid_index): {result}")
    
    return centroid_index, result

def centroid_corrector(x,centroid_index, centroid):
    i=j=k=0

    for row, col in enumerate(centroid_index):
        if (col==0):
            centroid[0] += centroid_index[row]
            i=i+1
        
        if (col==1):
            centroid[1] += centroid_index[row]
            j=j+1
        
        if (col==2):
            centroid[2] += centroid_index[row]
            k=k+1
        
    centroid[0] = centroid[0]/i
    centroid[1] = centroid[1]/j
    centroid[2] = centroid[2]/k
    return centroid


def once_more(x,centroid_index, centroid,x_reshaped):
    cluster(centroid,x_reshaped)
    centroid_corrector(x,centroid_index, centroid)
