import numpy as np
import pandas as pd

np.seterr(divide='ignore',invalid='ignore')

class GRNN:
    def __init__(self,sigma = 0.1):
        self.sigma = sigma

    def gaussian_kernal(self,x,c):
        return np.exp(-np.sum((x-c)**2)/(2*self.sigma**2))

    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        y_pred = np.zeros(len(X))
        for i,x in enumerate(X):
            kernels = np.array([self.gaussian_kernal(x,c) for c in self.X_train])
            y_pred[i] = np.sum(kernels*self.y_train)/np.sum(kernels)
        return y_pred

X_train = np.random.rand(100,7)
y_train = np.random.rand(100)

# X_train = np.array([[81,4,2,3,6,5,9],[98,5,4,6,5,1,3],[55,6,4,5,7,3,5],
#             [67,2,3,4,5,3,4],[89,6,7,5,6,7,8],[58,9,8,7,8,6,7],
#             [77,3,4,5,4,3,4],[95,1,3,2,1,2,0],[84,4,3,5,4,3,4],
#             [56,7,8,9,8,9,7],[45,3,5,4,3,4,5],[68,6,7,9,5,4,7],
#             [81,4,5,8,7,9,8],[91,6,9,8,7,8,9],[71,2,3,5,4,3,4],
#             [58,7,7,5,4,3,6],[59,7,8,6,9,10,6],[98,7,8,9,7,8,6],
#             [60,3,5,4,3,4,5],[88,5,7,6,4,8,6]])
# y_train = np.array([85,97,80,69,95,88,77,85,84,85,44,
#                      90,88,96,74,77,90,98,64,87])

# data0 = pd.read_excel('pre score training.xlsx')
# data1 = pd.read_excel('pre score stu.xlsx')
# X_train = data0.iloc[:,:7].values
# y_train = data0.iloc[:,7].values
# X_train = X_train.reshape(20,7)
# y_train = y_train.reshape(20)
# X_train,y_train = X_train/100.0,y_train/100.0  # 归一化处理

model = GRNN(sigma=0.1)
model.fit(X_train,y_train)
X_test =np.random.rand(1,7)
# X_test = np.array([80,4,5,7,6,5
# ,6])

# X_test = data1.iloc[:,:7].values
# X_test = X_test.reshape(1,7)
# X_test = X_test/100.0  # 归一化处理

pre = model.predict(X_test)
pre *= 100
print("pre_grade:",pre)

pre = float(pre)
pre *= 100
pre = int(pre)
pre /= 100
f = open('history pre_score.txt','a')
f1 = f.write(str(pre))
f2 = f.write(' ')
f.close()
