import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

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

# X_train = np.random.rand(100,7)
# y_train = np.random.rand(100)

# 读取训练集并进行训练
data0 = pd.read_excel('score training set.xlsx')
data1 = pd.read_excel('score stu set.xlsx')
X_train = data0.iloc[:,:7].values
y_train = data0.iloc[:,7].values
X_train,y_train = X_train/100.0,y_train/100.0

# X_train = np.array([[81,4,2,3,6,5,9],[98,5,4,6,5,1,3],[55,6,4,5,7,3,5],
#            [67,2,3,4,5,3,4],[89,6,7,5,6,7,8],[58,9,8,7,8,6,7],[77,3,4,5,4,3,4],
#            [95,1,3,2,1,2,0],[84,4,3,5,4,3,4],[56,7,8,9,8,9,7],[45,3,5,4,3,4,5],
#            [68,6,7,9,5,4,7],[81,4,5,8,7,9,8],[91,6,9,8,7,8,9],
#            [71,2,3,5,4,3,4],[58,7,7,5,4,3,6],[59,7,8,6,9,10,6],[98,7,8,9,7,8,6],
#            [60,3,5,4,3,4,5],[88,5,7,6,4,8,6]])
# y_train = np.array([85,97,80,69,95,88,77,85,84,85,44,90,88,96,74,77,90,98,64,87])

model = GRNN(sigma=0.1)
model.fit(X_train,y_train)

# 读取全班特征并开始预测
# X_test =np.random.rand(20,7)
X_test = data1.iloc[:,:7].values
X_test = X_test/100.0
# X_test = np.array([80,5,7,6,4,5,6])
pre = model.predict(X_test)
pre *= 100
print("pre_grade:",pre)
grade = pre

# 计算并输出各种属性
ave = np.mean(pre)
max0 = np.max(pre)
min0 = np.min(pre)
max_man = [i+1 for i, x in enumerate(pre) if x == max0]
min_man = [i+1 for i, x in enumerate(pre) if x == min0]
for i in max_man:
    max_m = i
for j in min_man:
    min_m = j

print('班级平均分：',ave)
print(f'班级最高分{max0},获得者{data1.iloc[max_m-1,7]}')
print(f'班级最低分{min0},获得者{data1.iloc[min_m-1,7]}')

# 导出为excel
list0 = data1.iloc[:,7]
writer = pd.ExcelWriter('全班预测成绩表.xlsx')
list1 = []
list1.append(pre)
test1 =pd.DataFrame(columns=list0,data=list1)
list2 = []
column = ['ave','max','min']
l0 = [ave,max0,min0]
list2.append(l0)
test2 = pd.DataFrame(columns=column,data=list2)
test1.to_excel(writer,sheet_name='预测学生成绩',index=False)
test2.to_excel(writer,sheet_name='平均分，最高分与最低分',index=False)
writer.close()

# 存档历史ave，并保留两位小数
ave *= 100
ave = int(ave)
ave /= 100
f = open('history ave score.txt','a')
f1 = f.write(str(ave))
f2 = f.write(' ')
f.close()

# 存档历史max，并保留两位小数
max0 *= 100
max0 = int(max0)
max0 /= 100
f = open('history max score.txt','a')
f1 = f.write(str(max0))
f2 = f.write(' ')
f.close()

# 存档历史min，并保留两位小数
min0 *= 100
min0 = int(min0)
min0 /= 100
f = open('history min score.txt','a')
f1 = f.write(str(min0))
f2 = f.write(' ')
f.close()

plt.rcParams['font.sans-serif']='SimHei'
plt.hist(grade,bins=20,edgecolor = 'black')
plt.xlabel('成绩')
plt.ylabel('学生人数')
plt.title('学生成绩直方图')
plt.grid(True)  # 显示网格线
save_path = "pictures"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path,'班级成绩预测直方图.png'))
plt.show()
