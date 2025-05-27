import matplotlib.pyplot as plt
import os
from matplotlib.ticker import MultipleLocator

f = open('historyT fail num.txt','r')
f1 = f.read()
f2 = str.split(f1)
x = []
for i in f2:
    i = float(i)
    x.append(i)
print(x)
f.close()

f = open('historyT gen num.txt','r')
f1 = f.read()
f2 = str.split(f1)
y = []
for i in f2:
    i = float(i)
    y.append(i)
print(y)
f.close()

f = open('historyT good num.txt','r')
f1 = f.read()
f2 = str.split(f1)
z = []
for i in f2:
    i = float(i)
    z.append(i)
print(z)
f.close()

plt.rcParams['font.sans-serif']='SimHei'
plt.figure(figsize=(8,8))
plt.subplot(2,2,1)
plt.plot(x,label = 'fail num',marker = "o")
plt.plot(y,label = 'general num',marker = "s")
plt.gca().axes.get_xaxis().set_visible(False)
plt.title('不及格人数与一般生人数走势图',font = {'family':'SimHei','size':14},loc = 'left')
plt.rc('font',family='Times New Roman')
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
for i, j in enumerate(x):
    plt.text(i, j+0.005, j, ha='center', va='bottom')
for i, j in enumerate(y):
    plt.text(i, j+0.005, j, ha='center', va='bottom')
plt.legend()

plt.subplot(2,2,2)
plt.plot(x,label = 'fail num',marker = "o")
plt.plot(z,label = 'good num',marker = "*")
plt.gca().axes.get_xaxis().set_visible(False)
plt.title('不及格人数与优秀生人数走势图',font = {'family':'SimHei','size':14},loc = 'left')
plt.rc('font',family='Times New Roman')
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
for i, j in enumerate(x):
    plt.text(i, j+0.005, j, ha='center', va='bottom')
for i, j in enumerate(z):
    plt.text(i, j+0.005, j, ha='center', va='bottom')
plt.legend()

plt.subplot(2,2,3)
plt.plot(y,label = 'general num',marker = "s")
plt.plot(z,label = 'good num',marker = "*")
plt.gca().axes.get_xaxis().set_visible(False)
plt.title('一般生人数与优秀生人数走势图',font = {'family':'SimHei','size':14},loc = 'left')
plt.rc('font',family='Times New Roman')
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
for i, j in enumerate(y):
    plt.text(i, j+0.005, j, ha='center', va='bottom')
for i, j in enumerate(z):
    plt.text(i, j+0.005, j, ha='center', va='bottom')
plt.legend()

plt.subplot(2,2,4)
plt.plot(x,label = 'fail num',marker = "o")
plt.plot(y,label = 'general num',marker = "s")
plt.plot(z,label = 'good num',marker = "*")
plt.gca().axes.get_xaxis().set_visible(False)
plt.title('综合评估走势图',font = {'family':'SimHei','size':14},loc = 'left')
plt.rc('font',family='Times New Roman')
x_major_locator=MultipleLocator(1)
ax=plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
for i, j in enumerate(x):
    plt.text(i, j+0.005, j, ha='center', va='bottom')
for i, j in enumerate(y):
    plt.text(i, j+0.005, j, ha='center', va='bottom')
for i, j in enumerate(z):
    plt.text(i, j+0.005, j, ha='center', va='bottom')
plt.legend()

plt.subplots_adjust(wspace=0.3, hspace=0.3)

save_path = "pictures"
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(os.path.join(save_path,'综评折线图.png'))

plt.show()