# 请确保您导入的权重矩阵正确，否则将对结果准确性产生影响！

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 输入目标个数
class_objective = int(input('请输入课程目标个数：'))  # 课程目标数 5
assessment_method = int(input("请输入考核方式数："))  # 考核方式数 6
indicator_points = int(input("请输入指标点个数："))   # 指标点个数 3
n = int(input("输入显示小数点位数："))

# 读入考核方式学生数据
fail_path = '考核方式全班分数表.xlsx'
data0 = pd.read_excel(fail_path)
score = data0.iloc[:,1:assessment_method+1].values
ave_score = np.mean(score,axis=0)
ave_score = ave_score.reshape(1,assessment_method)   # 考核平均分矩阵

# 导入考核方式对应课程目标权重矩阵
fail_path1 = "考核方式对应课程目标权重矩阵.xlsx"
data1 = pd.read_excel(fail_path1)
w1 = data1.iloc[:,1:class_objective+1].values
w1 = w1.reshape(assessment_method,class_objective)
if w1.shape[0]!=assessment_method:print("注意：导入矩阵行数与考核方式数不符，计算结果有可能不准确！")
if w1.shape[1]!=class_objective:print("注意：导入矩阵列数与课程目标数不符，计算结果有可能不准确！")

# 导入课程目标对应指标点权重矩阵
fail_path2 = "课程目标对应指标点权重矩阵.xlsx"
data2 = pd.read_excel(fail_path2)
w2 = data2.iloc[:,1:indicator_points+1].values
w2 = w2.reshape(class_objective,indicator_points)
if w2.shape[0]!=class_objective:print("注意：导入矩阵行数与课程目标数不符，计算结果有可能不准确！")
if w2.shape[1]!=indicator_points:print("注意：导入矩阵列数与指标点数不符，计算结果有可能不准确！")

# 导入指标点权重矩阵
fail_path3 = "指标点权重矩阵.xlsx"
data3 = pd.read_excel(fail_path3)
w3 = data3.iloc[:,:indicator_points].values
w3 = w3.reshape(1,indicator_points)
if w3.shape[1]!=indicator_points:print("注意：导入矩阵列数与指标点数不符，计算结果有可能不准确！")

# 计算课程目标达成度
class_achievement = np.dot(ave_score,w1)
class_achievement = class_achievement.reshape(1,class_objective)
class_achievement = np.around(class_achievement,decimals=n)

# 计算指标点达成度
indicator_points_achievement = np.dot(class_achievement,w2)
indicator_points_achievement = indicator_points_achievement.reshape(1,indicator_points)
indicator_points_achievement = np.around(indicator_points_achievement,decimals=n)

# 计算指标点评价值
indicator_points_value = indicator_points_achievement*w3
indicator_points_value = indicator_points_value.reshape(1,indicator_points)
indicator_points_value = np.around(indicator_points_value,decimals=n)

# 计算课程达成度评价值
class_final = np.mean(indicator_points_value)
class_final*=100
class_final=int(class_final)
class_final/=100

# 矩阵重塑
class_achievement = class_achievement.reshape(class_objective)
indicator_points_achievement = indicator_points_achievement.reshape(indicator_points)
indicator_points_value = indicator_points_value.reshape(indicator_points)

# 输出重塑后矩阵
print("课程目标达成度：")
print(class_achievement)
print("指标点达成度：")
print(indicator_points_achievement)
print("指标点评价值：")
print(indicator_points_value)
print("课程达成度评价值")
print(class_final)

# 历史记录写入
f = open('history A.txt','a')
f1 = f.write(str(class_final))
f2 = f.write(' ')
f.close()

# 生成并导出分析图：准备工作
list1 = []
list2 = []
list3 = []
for i in class_achievement:
    list1.append(i)

for j in indicator_points_achievement:
    list2.append(j)

for k in indicator_points_value:
    list3.append(k)

# 获取excel列标题
title1 = data1.columns.tolist()
x1 = title1[1:class_objective+1]
title2 = data2.columns.tolist()
x2 = title2[1:indicator_points+1]
title3 = data3.columns.tolist()
x3 = title3[:indicator_points]

# 生成图片1课程目标达成度图
plt.rcParams['font.sans-serif']='Microsoft YaHei'
plt.bar(x1, list1, color='skyblue', width=0.4, edgecolor='black')
plt.title('课程目标达成度', fontsize=16, color='blue')
plt.xlabel('课程目标', fontsize=12, color='green')
plt.ylabel('达成度', fontsize=12, color='purple')
plt.grid(True)
plt.legend(['课程目标达成度'], loc='lower right')
for i, j in enumerate(list1):
    plt.text(i, j+0.5, str(j), ha='center')
save_path = "pictures"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path,'课程目标达成度图.png'))
plt.show()

# 生成图片2指标点达成度图
plt.rcParams['font.sans-serif']='Microsoft YaHei'
plt.bar(x2, list2, color='hotpink', width=0.4, edgecolor='black')
plt.title('指标点达成度', fontsize=16, color='blue')
plt.xlabel('指标点', fontsize=12, color='green')
plt.ylabel('达成度', fontsize=12, color='purple')
plt.grid(True)
plt.legend(['指标点达成度'], loc='lower right')
for i, j in enumerate(list2):
    plt.text(i, j+0.5, str(j), ha='center')
save_path = "pictures"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path,'指标点达成度图.png'))
plt.show()

# 生成图片3指标点评价值图
plt.rcParams['font.sans-serif']='Microsoft JhengHei'
plt.bar(x3, list3, color='lightgreen', width=0.4, edgecolor='black')
plt.title('指标点评价值', fontsize=16, color='blue')
plt.xlabel('指标点', fontsize=12, color='green')
plt.ylabel('评价值', fontsize=12, color='purple')
plt.grid(True)
plt.legend(['指标点评价值'], loc='lower right')
for i, j in enumerate(list3):
    plt.text(i, j+0.5, str(j), ha='center')
save_path = "pictures"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path,'指标点评价值图.png'))
plt.show()


# 代码核心原理
# m = 4
# n = 5
# k = 3
# w1 = 100*np.random.rand(1,m)
# w2 = np.random.rand(m,n)
# w3 = np.random.rand(n,k)
# w4 = np.random.rand(1,k)
#
# w12 = np.dot(w1,w2)
# w123 = np.dot(w12,w3)
# w1234 = w123*w4
#
# w12 = np.around(w12, decimals=2)
# w123 = np.around(w123, decimals=2)
# w1234 = np.around(w1234, decimals=2)
# print(w12)
# print(w123)
# print(w1234)
