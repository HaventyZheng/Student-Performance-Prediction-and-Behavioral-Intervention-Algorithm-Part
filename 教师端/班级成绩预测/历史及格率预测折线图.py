import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator

f = open('history pass rate.txt','r')
f1 = f.read()
f2 = str.split(f1)
x = []
for i in f2:
    i = float(i)
    x.append(i)
print(x)
f.close()

plt.plot(x,marker = "o")
plt.gca().axes.get_xaxis().set_visible(False)
plt.title("历史预测及格率折线图",font = {'family':'SimHei','size':14},loc = 'left' )  # 图标名称
plt.xlabel("预测次数",font = {'family':'FangSong','size':13})  # x轴名称
plt.ylabel("预测及格率",font = {'family':'FangSong','size':13})  # y轴名称
plt.rc('font',family='Times New Roman')
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
for i, j in enumerate(x):
    plt.text(i, j+0.005, j, ha='center', va='bottom')
save_path = "pictures"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path,'历史预测及格率折线图.png'))
plt.show()