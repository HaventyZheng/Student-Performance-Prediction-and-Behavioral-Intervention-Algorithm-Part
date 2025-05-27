import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn import tree
import graphviz
from IPython.display import Image
import pydotplus

numb = int(input())

# 读取数据从excel中，划分训练集与测试集
data0 = pd.read_excel('features&labels.xlsx')
data1 = pd.read_excel('stu f&l.xlsx')
data0.iloc[:, numb] = data0.iloc[:, numb].apply(lambda x: 0 if x < 60 else x)
data0.to_excel('features&labels.xlsx',index=False)
data0.iloc[:, numb] = data0.iloc[:, numb].apply(lambda x: 1 if 60 <= x < 90 else x)
data0.to_excel('features&labels.xlsx',index=False)
data0.iloc[:, numb] = data0.iloc[:, numb].apply(lambda x: 2 if x >= 90 else x)
data0.to_excel('features&labels.xlsx',index=False)
data0 = pd.read_excel('features&labels.xlsx')
features = data0.iloc[:,:numb].values
labels = data0.iloc[:,numb].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 建立决策树模型
model = DecisionTreeClassifier()

# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上做预测
predictions = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# 读取新的数据并使用模型预测新的学生数据
student = data1.iloc[:,:numb].values
predictions = model.predict(student)

# 分析及格,一般，优秀人数
counts = np.bincount(predictions)
num_fail = counts[0]
num_gen = counts[1]
num_good = counts[2]
fail_rate = num_fail/len(predictions)
gen_rate = (len(predictions) - num_good - num_fail)/len(predictions)
good_rate = num_good/len(predictions)

# 输出各项指标
print('num_fail:',num_fail)
print('num_gen:',num_gen)
print('num_good',num_good)
print('fail_rate',fail_rate)
print('gen_rate:',gen_rate)
print('good_rate:',good_rate)

# # 生成决策树图
# dot_data = tree.export_graphviz(
#     model,
#     feature_names=['考试成绩','出勤率','作业完成情况','视频重复观看次数','视频观看时长','作业正确率','平时小测平均分'],
#     filled=True,
#     rounded=True
# )
# graph = graphviz.Source(dot_data)
#
# # 保存决策树图
# dot_data = dot_data.replace('helvetica', 'MicrosoftYaHei')
#
# graph = pydotplus.graph_from_dot_data(dot_data)
# img = Image(graph.create_png())
# graph.write_png('D:/大学文件/计算机编程实验与网页设计实验/python编程实验/08 大创项目算法/pythonProject1/教师端/班级成绩预测/pictures/综评预测.png')

# 保存历史数据
f = open('historyT fail num.txt','a')
f1 = f.write(str(num_fail))
f2 = f.write(' ')
f.close()

f = open('historyT gen num.txt','a')
f1 = f.write(str(num_gen))
f2 = f.write(' ')
f.close()

f = open('historyT good num.txt','a')
f1 = f.write(str(num_good))
f2 = f.write(' ')
f.close()

# 生成并保存总评饼图
plt.rcParams['font.sans-serif']='SimHei'
plt.figure(figsize=(6,6))
label=['不及格','一般','优秀']
explode=[0.03,0.01,0.01]
values=[num_fail,num_gen,num_good]
plt.pie(values,explode=explode,labels=label,autopct='%1.1f%%',shadow=True)
plt.title('综合情况饼图')
plt.legend(loc = 'right')
save_path = "pictures"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path,'综评饼图.png'))
plt.show()