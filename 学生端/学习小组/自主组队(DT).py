from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import graphviz
from IPython.display import Image
import pydotplus
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

data0 = pd.read_excel('数据库学生特征.xlsx')

f_num = data0.iloc[:,:5].values
f_str = data0.iloc[:,5].values
labels = data0.iloc[:,7].values

# print(f_num)
# print()
# print(f_str)
# print()
# print(labels)
# print()

vectorizer = TfidfVectorizer()
tfidf_data = vectorizer.fit_transform(f_str)
# print(tfidf_data)
# print()
features = np.concatenate((f_num, tfidf_data.toarray()), axis=1)
# print(features)

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
data1 = pd.read_excel('待分组学生.xlsx')
new_num = data1.iloc[:,:5].values
new_str = data1.iloc[:,5].values
all_str = np.append(f_str,new_str)
all_num = np.append(f_num,new_num,axis=0)
# print(all_num)
# print(all_str)
# print()
# print(new_num)
# print()
# print(new_str)
# print()
vectorizer0 = TfidfVectorizer()
tfidf_data0 = vectorizer0.fit_transform(all_str)
# last_row = tfidf_data0[-1]
# print(last_row)
all_student = np.concatenate((all_num, tfidf_data0.toarray()), axis=1)
# print()
# print(all_student)
new_student = all_student[-1]
new_student = new_student.reshape(1,-1)
# print(new_student)
prediction = model.predict(new_student)

print(f'分到第{prediction[0]}组')

old_c = data0['Cluster'].values
old_name = data0['学号'].values
group_index = np.where(old_c==prediction[0])
group_name = old_name[group_index]
print(group_name)

list1 = []
list1.append(group_name)
test1 = pd.DataFrame(list1).T
test1.columns = ['小组成员']
test1.to_excel('小组成员导出名单.xlsx',index=False)

dot_data = tree.export_graphviz(
    model,
    feature_names=['学习时长','是否熬夜学习','是否早起','习惯自学','习惯他授','one_hot1','one_hot2','one_hot3','one_hot4','one_hot5','one_hot6','one_hot7','one_hot8'],
    filled=True,
    rounded=True
)
graph = graphviz.Source(dot_data)  # 直接输出
# print(graph)

dot_data = dot_data.replace('helvetica', 'MicrosoftYaHei')

graph = pydotplus.graph_from_dot_data(dot_data)
img = Image(graph.create_png())
graph.write_png('D:/大学文件/计算机编程实验与网页设计实验/python编程实验/08 大创项目算法/pythonProject1/学生端/学习小组/pictures/小组分配决策树图.png')