import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
font_location = "C:/Windows/Fonts/HANDotum.ttf"
font_name = fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font', family=font_name)

npoints = 1000
vectors = np.zeros([npoints,2])
for i in range(npoints):
    np.random.seed(i)
    if np.random.uniform()>0.5:
        vectors[i,0] = np.random.normal(0.0, 1.0)
        vectors[i,1] = np.random.normal(0.0, 1.0)
    else:
        vectors[i,0] = np.random.normal(4.0, 0.5)
        vectors[i,1] = np.random.normal(2.0, 0.5)

k=2
centroids = np.zeros([k,2])
for i in range(k):
    np.random.seed(i)
    centroids[i,0]=np.random.normal(0.0, 0.5)*(i*2+1)
    centroids[i,1]=np.random.normal(0.0, 0.5)*(-i*2+1)

tmp = np.expand_dims(np.sum(np.square(vectors - centroids[0,:]),axis=1),axis=1)
tmp2 = np.expand_dims(np.sum(np.square(vectors - centroids[1,:]),axis=1),axis=1)
tmp3 = np.append(tmp,tmp2,axis=1)
assignments = np.argmin(tmp3,axis=1)

plt.figure(1)
for i in range(npoints):
    if assignments[i]==0:
        plt.scatter(vectors[i,0], vectors[i,1], marker = '.',  c = 'b')
    else:
        plt.scatter(vectors[i,0], vectors[i,1], marker = '.',  c = 'r')
        
plt.scatter(centroids[0,0], centroids[0,1], marker = '^', c = 'k', label = 'center1')
plt.scatter(centroids[1,0], centroids[1,1], marker = 'o', c = 'g', label = 'center2')
plt.legend()
plt.title("<평균 업데이트 전>")

####################################################
mask1 = np.where(assignments==0)
mask2 = np.where(assignments==1)

centroids[0,:] = np.mean(vectors[mask1,:],axis=1)
centroids[1,:] = np.mean(vectors[mask2,:],axis=1)

tmp = np.expand_dims(np.sum(np.square(vectors - centroids[0,:]),axis=1),axis=1)
tmp2 = np.expand_dims(np.sum(np.square(vectors - centroids[1,:]),axis=1),axis=1)
tmp3 = np.append(tmp,tmp2,axis=1)
assignments = np.argmin(tmp3,axis=1)

plt.figure(2)
for i in range(npoints):
    if assignments[i]==0:
        plt.scatter(vectors[i,0], vectors[i,1], marker = '.',  c = 'b')
    else:
        plt.scatter(vectors[i,0], vectors[i,1], marker = '.',  c = 'r')
        
plt.scatter(centroids[0,0], centroids[0,1], marker = '^', c = 'k', label = 'center1')
plt.scatter(centroids[1,0], centroids[1,1], marker = 'o', c = 'g', label = 'center2')
plt.legend()
plt.title("<평균 업데이트 후>")
plt.show() 
