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

numb = int(input("特征个数："))  # 6

# 读取数据从excel中，划分训练集与测试集
data0 = pd.read_excel('features&labels good.xlsx')
data0.iloc[:, numb] = data0.iloc[:, numb].apply(lambda x: 0 if x < 90 else x)
data0.to_excel('features&labels good.xlsx',index=False)
data0.iloc[:, numb] = data0.iloc[:, numb].apply(lambda x: 1 if x >= 90 else x)
data0.to_excel('features&labels good.xlsx',index=False)
data0 = pd.read_excel('features&labels good.xlsx')
data1 = pd.read_excel('stu f&l good.xlsx')
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

# 分析优秀人数，优秀率，优秀名单
num_good = sum(predictions)
num_gen = len(predictions) - num_good
good_rate = num_good/len(predictions)
index_of_good = [i for i,p in enumerate(predictions) if p]
code_of_good = [i+1 for i in index_of_good]

# 输出各项指标
print('good_rate:',good_rate)
print('num_good:',num_good)
print('num_gen',num_gen)
list0 = data1.iloc[index_of_good,numb]
print(index_of_good)
print(code_of_good)
print('code of good:',list0)

# 生成并导出优秀成员名单
data2 = {'优秀名单':list0}
test0 = pd.DataFrame(data2)
test0.to_excel('优秀预测成员表.xlsx',sheet_name='预测优秀学生',index=False)

# # 生成决策树图
# dot_data = tree.export_graphviz(
#     model,
#     feature_names=[data1[1],data1[2],data1[3],data1[4],data1[5],data1[6]],
#     filled=True,
#     rounded=True
# )
# graph = graphviz.Source(dot_data)
#
# # 保存决策树图
# dot_data = dot_data.replace('helvetica', 'MicrosoftYaHei')

# graph = pydotplus.graph_from_dot_data(dot_data)
# img = Image(graph.create_png())
# graph.write_png('D:/大学文件/计算机编程实验与网页设计实验/python编程实验/08 大创项目算法/pythonProject1/教师端/班级成绩预测/pictures/优秀预测.png')

# 保存历史数据
f = open('history good rate.txt','a')
f1 = f.write(str(good_rate))
f2 = f.write(' ')
f.close()

# 生成并保存优秀率饼图
plt.rcParams['font.sans-serif']='SimHei'
plt.figure(figsize=(6,6))
label=['优秀','一般']
explode=[0.01,0.01]
values=[num_good,num_gen]
plt.pie(values,explode=explode,labels=label,autopct='%1.1f%%',shadow=True)
plt.title('优秀情况饼图')
plt.legend(['优秀','一般'],title = '图例',loc = 'right')
save_path = "pictures"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path,'优秀率饼图.png'))
plt.show()