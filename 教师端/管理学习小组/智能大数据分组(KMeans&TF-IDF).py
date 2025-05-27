import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# 从Excel文件中读取学生数据
student_data = pd.read_excel('学生特征.xlsx')

# 将特征数据转换为numpy数组
student_array = student_data.values

# 取出所有特征，除了第六个特征（假设第六个特征是字符串型），以及第七列（学生姓名）
numeric_data = student_array[:, :5]  # 前五个特征为数值型特征
string_data = student_array[:, 5]  # 第六个特征为字符串型特征
student_codes = student_array[:, 6]  # 学生姓名在第七列

# 使用TfidfVectorizer对字符串型特征进行TF-IDF处理
vectorizer = TfidfVectorizer()
tfidf_data = vectorizer.fit_transform(string_data)

# 将数值特征和TF-IDF处理后的特征合并
combined_data = np.concatenate((numeric_data, tfidf_data.toarray()), axis=1)

# 使用StandardScaler对数据进行标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(combined_data)

# 交叉验证以确定最优的K值
k_values = range(2, 20)  # 假设尝试K值从2到9
best_k = 0

test1 = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(scaled_data)
    y_pred = kmeans.labels_
    s = silhouette_score(scaled_data,y_pred)
    s *=100
    s=int(s)
    s/=100
    test1.append(s)
lst = [0 for i in range(100)]
max_s = np.max(test1)
for i in range(2,20):
    lst[i] = test1[i-2]
for i in range(2,20):
    if lst[i]>lst[best_k]:
        best_k = i

# 使用最优K值进行学生分组
kmeans = KMeans(n_clusters=best_k, random_state=0)
kmeans.fit(scaled_data)

student_data['小组'] = [i+1 for i in kmeans.labels_]

# 打印每个小组的学生列表
groups = kmeans.labels_
group_students = {}

for group in range(best_k):
    group_students[group] = []

for idx, student_name in enumerate(student_codes):
    group = groups[idx]
    group_students[group].append(student_name)
print(f'共分成{best_k}组')

for group, students in group_students.items():
    students = str(students)
    print("学习小组{}的学生: {}".format(group + 1, ''.join(students)))

file_path = input('请输入导出Excel文件名:')
file_path = file_path+'.xlsx'
student_data.to_excel(file_path,index=False)