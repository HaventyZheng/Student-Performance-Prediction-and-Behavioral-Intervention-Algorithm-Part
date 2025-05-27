# 在处理Excel文件时，确实有一些存储格式的要求，以确保程序能够正确读取数据。以下是一些建议的存储格式要求：
# 1.数据表格格式 ：确保数据以表格形式存储在Excel文件中，每一列代表一个特征，每一行代表一个学生的数据。
# 2.列标题 ：每一列应该有清晰的列标题，用于标识该列数据的含义，例如“学生姓名”、“学习时长”、“是否熬夜学习”等。
# 3.数据类型 ：确保每一列的数据类型是一致的，例如学习时长应该是数值型数据，是否熬夜学习应该是布尔型数据等。
# 4.缺失值处理 ：如果有缺失值，可以选择填充缺失值或者删除包含缺失值的行，确保数据完整性。
# 5.数据范围 ：确保数据的范围合理，不要包含异常值或超出范围的数据。
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

k = int(input('请输入分组组数:'))
# 从 Excel 文件中读取学生信息
df = pd.read_excel("学生特征.xlsx")

# 提取特征列
features = df[['学习时长', '是否熬夜学习', '是否早起', '习惯自学', '习惯他授', '兴趣爱好']]

# 对兴趣爱好进行 TF-IDF 向量化处理
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(features['兴趣爱好'])

# 数值特征标准化
scaler = StandardScaler()
X_numeric = scaler.fit_transform(features[['学习时长', '是否熬夜学习', '是否早起', '习惯自学', '习惯他授']])

# 合并文本特征和数值特征
X_combined = np.hstack((X_text.toarray(), X_numeric))

# 使用 K-means 聚类算法将学生分为3类
kmeans = KMeans(n_clusters=k, random_state=0) # n_cluster为K值
kmeans.fit(X_combined)

# 将聚类结果添加到原始数据中
df['Cluster'] = [i+1 for i in kmeans.labels_]

# 输出每个学生所属的类别
for index, row in df.iterrows():
    print(f"学生 {index+1} -> 学习小组 {row['Cluster']+1}")

# 输出每组所含学生姓名
groups = kmeans.labels_
group_students = {}
for i in range(k):
    group_students[i] = []
student_names = df.iloc[:, 6]
for idx, student_name in enumerate(student_names):
    group = groups[idx]
    group_students[group].append(student_name)
for group, students in group_students.items():
    students = str(students)
    print("学习小组{}的学生: {}".format(group + 1, ''.join(students)))

# 导出excel文件
file_path = input('请为导出的Excel文件命名：')
file_path = file_path + '.xlsx'
df.to_excel(file_path,index=False)



