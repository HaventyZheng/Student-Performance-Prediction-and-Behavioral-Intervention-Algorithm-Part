from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz
from IPython.display import Image
import pydotplus
import pandas as pd

# features = [
#     [80, 90, 95,6,56,78,45],
#     [70, 80, 75,5,67,34,56],
#     [60, 50, 70,2,45,33,45],
#     # 添加更多学生的特征数据
# ]
# labels = [1, 1, 0]  # 学生1、2可能及格，学生3可能不及格

numb = int(input())

# 从excel中导入训练集,excel的第一行不算数
data0 = pd.read_excel('training set pass.xlsx')

data0.iloc[:, numb] = data0.iloc[:, numb].apply(lambda x: 0 if x < 60 else x)
data0.to_excel('training set pass.xlsx',index=False)
data0.iloc[:, numb] = data0.iloc[:, numb].apply(lambda x: 1 if x >= 60 else x)
data0.to_excel('training set pass.xlsx',index=False)
data0 = pd.read_excel('training set pass.xlsx')

features = data0.iloc[:,:numb].values
labels  = data0.iloc[:,numb].values

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# 建立决策树模型
model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=30,
    splitter="random"
    # max_depth=3,
    # min_samples_split=10,
    # min_samples_leaf=5,
    # max_features=7,
    # min_impurity_decrease=
)

# 在训练集上训练模型
model.fit(X_train, y_train)

# 在测试集上做预测
predictions = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# 使用模型预测新的学生数据
data1 = pd.read_excel('stu set pass.xlsx')
# new_student = [[65, 70, 80,3,50,33,45]]  # 新学生的特征数据
new_student = data1.iloc[:,:numb].values
prediction = model.predict(new_student)
if prediction[0] == 1:
    print("该学生可能及格")
else:
    print("该学生可能不及格")

# dot_data = tree.export_graphviz(
#     model,
#     feature_names=['考试成绩','出勤率','作业完成情况','视频重复观看次数','视频观看时长','作业正确率','平时小测平均分'],
#     filled=True,
#     rounded=True
# )
# graph = graphviz.Source(dot_data)  # 直接输出
# print(graph)
#
# dot_data = dot_data.replace('helvetica', 'MicrosoftYaHei')
#
# graph = pydotplus.graph_from_dot_data(dot_data)
# img = Image(graph.create_png())
# graph.write_png('D:/大学文件/计算机编程实验与网页设计实验/python编程实验/08 大创项目算法/pythonProject1/学生端/课程成绩预测/pictures/及格预测.png')

# feature_name = ['考试成绩','出勤率','作业完成情况','视频重复观看次数','视频观看时长','作业正确率','平时小测平均分']

# model.feature_importances_
# print([*zip(feature_name,model.feature_importances_)])