from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

k = int(input('请输入分组组数:'))
# 从 Excel 文件中读取学生信息
df = pd.read_excel("学生特征.xlsx")
df2 = pd.read_excel("学生特征.xlsx")

# 提取特征列
features = df[['学习时长', '是否熬夜学习', '是否早起', '习惯自学', '习惯他授', '兴趣爱好']]

# 对兴趣爱好进行 TF-IDF 向量化处理
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(features['兴趣爱好'])

# 归一化
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features[['学习时长', '是否熬夜学习', '是否早起', '习惯自学', '习惯他授']])

# 标准化
scaler = StandardScaler()
standardized_features = scaler.fit_transform(features[['学习时长', '是否熬夜学习', '是否早起', '习惯自学', '习惯他授']])

# 合并文本特征和数值特征
normalized_combined = np.hstack((X_text.toarray(), normalized_features))
standardized_combined = np.hstack((X_text.toarray(), standardized_features))

# 聚类算法
kmeans = KMeans(n_clusters=k)  # 将学生分成k个小组

# 使用归一化后的特征进行聚类
kmeans.fit(normalized_combined)
labels_normalized = kmeans.labels_
df['小组'] = [i+1 for i in kmeans.labels_]

# 使用标准化后的特征进行聚类
kmeans.fit(standardized_combined)
labels_standardized = kmeans.labels_
df2['小组'] = [i+1 for i in kmeans.labels_]

# 输出
print("方案一：使用归一化后的特征Grouping using normalized features:")
group_students = {}
for i in range(k):
    group_students[i] = []
student_codes = df.iloc[:, 6]
for idx, student_name in enumerate(student_codes):
    student_name = str(student_name)
    group = labels_normalized[idx]
    group_students[group].append(student_name)
for group, students in group_students.items():
    print("学习小组{}的学生: {}".format(group + 1, ', '.join(students)))

print()

print("方案二：使用标准化后的特征Grouping using standardized features:")
group_students = {}
for i in range(k):
    group_students[i] = []
student_codes = df.iloc[:, 6]
for idx, student_name in enumerate(student_codes):
    student_name = str(student_name)
    group = labels_standardized[idx]
    group_students[group].append(student_name)
for group, students in group_students.items():
    print("学习小组{}的学生: {}".format(group + 1, ', '.join(students)))

# 导出为excel文件
df.to_excel('多方案归一化.xlsx',index=False)
df2.to_excel('多方案标准化.xlsx',index=False)
