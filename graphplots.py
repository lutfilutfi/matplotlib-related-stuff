import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.decomposition import PCA
# rng=np.random.RandomState(1)
# X=np.dot(rng.rand(2,2),rng.randn(2,200)).T
# print(rng.rand(2,2))
# plt.scatter(X[:,0],X[:,1])
# plt.axis('equal')
# plt.show()
x=np.linspace(0,10,25)
y=x*x+2
print(np.array([x,y]).reshape(25,2).T)
pylab.subplot(1,2,1)
pylab.plot(x,y,'r--')
pylab.subplot(1,2,2)
pylab.plot(y,x,'g*--')
pylab.show()
