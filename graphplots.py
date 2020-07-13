import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
from sklearn.decomposition import PCA
#imports random numbers and then plots them 
# rng=np.random.RandomState(1)
# X=np.dot(rng.rand(2,2),rng.randn(2,200)).T#doing dot product between 2 matrices
# print(rng.rand(2,2))
# plt.scatter(X[:,0],X[:,1])
# plt.axis('equal')
# plt.show()
#below is a way to plot polynomial graphs
x=np.linspace(0,2000,100)
y=x*x+-1000*x-2               #play with these to create different polynomials
# print(np.array([x,y]).reshape(25,2).T)
# pylab.subplot(1,2,1)
# pylab.plot(x,y,'r--')
# pylab.subplot(1,2,2)
# pylab.plot(y,x,'g*--')
# pylab.show()
# fig=plt.figure()    #fig is now the canvas
# axis=fig.add_axes([0.1,0.1,.9,.9])#left ,right indexing,height and widht of the canvas
# axis.plot(x,y,'r')
# plt.show()
fig,axes=plt.subplots(nrows=1,ncols=1)
print(fig)
# for ax in axes:
#     ax.scatter(x,y,c='r')
axes.scatter(x,y,c='r')
plt.show()