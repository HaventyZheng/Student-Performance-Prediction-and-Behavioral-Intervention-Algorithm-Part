import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import os

min_num = int(input('请输入最小分组数：'))
max_num = int(input('请输入最大分组数：'))
# 从Excel文件中读取学生数据
student_data = pd.read_excel('学生特征.xlsx')

# 将特征数据转换为numpy数组
student_array = student_data.values

# 取出所有特征，除了第六个特征（假设第六个特征是字符串型），以及第七列（学生姓名）
numeric_data = student_array[:, :5]  # 前五个特征为数值型特征
string_data = student_array[:, 5]  # 第六个特征为字符串型特征
student_names = student_array[:, 6]  # 学生姓名在第七列

# 使用TfidfVectorizer对字符串型特征进行TF-IDF处理
vectorizer = TfidfVectorizer()
tfidf_data = vectorizer.fit_transform(string_data)

# 将数值特征和TF-IDF处理后的特征合并
combined_data = np.concatenate((numeric_data, tfidf_data.toarray()), axis=1)

# 使用StandardScaler对数据进行标准化
scaler = StandardScaler()
scaled_data = scaler.fit_transform(combined_data)

# 交叉验证以确定最优的K值
k_values = range(min_num, max_num+1)  # 假设尝试K值从2到9
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
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.plot(range(min_num,max_num+1),test1,label = '优势程度（轮廓系数）')
plt.legend()
save_path = "pictures"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path,'超参数曲线.png'))
plt.show()

lst = [0 for i in range(100)]
max_s = np.max(test1)
for i in range(min_num,max_num+1):
    lst[i] = test1[i-min_num]
for i in range(min_num,max_num+1):
    if lst[i]>lst[best_k]:
        best_k = i
print('最优分组数为：',best_k)

