import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm

mat=loadmat("ex6data1.mat")
# print(mat.keys())

X=mat['X']
y=mat['y']
# print(len(y.flatten()))
def plotData(X,y):
    plt.figure(figsize=(8,5))
    plt.scatter(X[:,0],X[:,1],c=y.flatten(),cmap='rainbow')
    plt.xlabel('X1')
    plt.ylabel('X2')
    # plt.legend()
    # plt.show()
# plotData(X,y)

def plotBoundary(clf,X):
    x_min,x_max=X[:,0].min()*1.2,X[:,0].max()*1.1
    y_min,y_max=X[:,1].min()*1.2,X[:,1].max()*1.1
    xx,yy=np.meshgrid(np.linspace(x_min,x_max,500),
                      np.linspace(y_min,y_max,500))
    # print(xx.ravel().shape)
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    # print(Z.shape)
    Z=Z.reshape(xx.shape)
    # print("reshape之后："+str(Z.shape))
    # plt.contourf(xx,yy,Z)
    plt.contour(xx,yy,Z)
    plt.show()
models=[svm.SVC(C,kernel='linear') for C in [1,100]]
clfs=[model.fit(X,y.ravel()) for model in models]
title=['SVM Decision Boundary with C={} (Example Dataset 1'.format(C) for C in [1,100]]

# for model,title in zip(clfs,title):
#     plt.figure(figsize=(8,5))
#     plotData(X,y)
#     plotBoundary(model,X)
#     plt.title(title)
    # plt.show()

def gaussKernel(x1,x2,sigma):
    return np.float(np.exp(-((x1-x2) ** 2).sum()/(2.0 * (sigma ** 2))))

gaussKernel(np.array([1.0,2.0,1.0]),np.array([0,4,-1]),2.0)

mat=loadmat('ex6data2.mat')
X2=mat['X']
y2=mat['y']
# plotData(X2,y2)

# sigma=0.1
# gamma=np.power(sigma,-2.0)/2
# clf=svm.SVC(C=1,kernel='rbf',gamma=gamma)
# model=clf.fit(X2,y2.flatten())
# plotData(X2,y2)
# plotBoundary(model,X2)

mat3=loadmat('ex6data3.mat')
X3,y3=mat3['X'],mat3['y']
X_val,y_val=mat3['Xval'],mat3['yval']
# plotData(X_val,y_val)
# plt.show()

Cvalues=(0.01,0.03,0.1,0.3,1.,3.,10.,30.)
sigmavalues=Cvalues
best_pair,best_score=(0,0),0

for C in Cvalues:
    for sigma in sigmavalues:
        gamma=np.power(sigma,-2.)/2
        model=svm.SVC(C=C,kernel='rbf',gamma=gamma)
        model.fit(X3,y3.flatten())
        this_score=model.score(X_val,y_val)
        if this_score >best_score:
            best_score=this_score
            best_pair=(C,sigma)
# print('best_pair={},best_score={}'.format(best_pair,best_score))

model=svm.SVC(C=1.,kernel='rbf',gamma=np.power(.1,-2.)/2)
model.fit(X3,y3.flatten())
# plotData(X3,y3)
# plotBoundary(model,X3)

# y=np.array([0]*20+[1]*20)
# print(y[0])

np.random.seed(0)

X=np.array([[3,3],[4,3],[1,1]])
Y=np.array([1,1,-1])

clf=svm.SVC(kernel='linear')
clf.fit(X,Y)

# print(clf.coef_)
w=clf.coef_[0]
a=-w[0]/w[1]
xx=np.linspace(-5,5)
# print(clf.intercept_)
yy=a*xx-(clf.intercept_[0])/w[1]

print(clf.support_vectors_)
b=clf.support_vectors_[0]
yy_down=a*xx+(b[1]-a*b[0])
b=clf.support_vectors_[-1]
yy_up=a*xx+(b[1]-a*b[0])

plt.figure(figsize=(8,5))
plt.plot(xx,yy,'k-')
plt.plot(xx,yy_down,'k--')
plt.plot(xx,yy_up,'k--')

plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=150,facecolors='none',edgecolors='k',linewidths=1.5)
plt.scatter(X[:,0],X[:,1],c=Y,cmap=plt.cm.rainbow)

plt.axis('tight')
# plt.show()

# print(clf.decision_function(X))