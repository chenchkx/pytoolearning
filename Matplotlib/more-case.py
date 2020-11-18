# from: https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py



import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2, 100)

# Note that even in the OO-style, we use `.pyplot.figure` to create the figure.
fig, ax = plt.subplots()  # Create a figure and an axes.
ax.plot(x, x, label='linear')  # Plot some data on the axes.
ax.plot(x, x**2, label='quadratic')  # Plot more data on the axes...
ax.plot(x, x**3, label='cubic')  # ... and some more.
ax.set_xlabel('x label')  # Add an x-label to the axes.
ax.set_ylabel('y label')  # Add a y-label to the axes.
ax.set_title("Simple Plot")  # Add a title to the axes.
ax.legend()  # Add a legend(注释，解释).
plt.show()  # or : fig.show()


# 在两个区域画图，一个是曲线，一个是散点图。参考：
# https://blog.csdn.net/weixin_38753213/article/details/105885338
fig, ax = plt.subplots(1,2)  # Create a figure and two axes.
ax[0].plot(x, x, label='linear')  # Plot some data on the axes.
ax[0].plot(x, x**2, label='quadratic')  # Plot more data on the axes...
ax[0].plot(x, x**3, label='cubic')  # ... and some more.
ax[0].set_xlabel('x label')  # Add an x-label to the axes.
ax[0].set_ylabel('y label')  # Add a y-label to the axes.
ax[0].set_title("Simple Plot")  # Add a title to the axes.
ax[0].legend()  # Add a legend.
ax[1].scatter(x, x, label='linear')  # scatter some data on the axes.
ax[1].scatter(x, x**2, label='quadratic')  # scatter more data on the axes...
ax[1].scatter(x, x**3, label='cubic')  # ... and some more.
ax[1].set_xlabel('x label')  # Add an x-label to the axes.
ax[1].set_ylabel('y label')  # Add a y-label to the axes.
ax[1].set_title("Simple Plot")  # Add a title to the axes.
ax[1].legend()  # Add a legend.
plt.show() # or : fig.show()


# 画散点图也可以使用 plot，不过需要使用格式化字符 {'marker': 'x'}。参考：
# https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
# https://www.runoob.com/numpy/numpy-matplotlib.html
fig, ax = plt.subplots(1,2)  # Create a figure and two axes.
ax[0].plot(x, x, "ob", label='linear')  # Plot some data on the axes.
ax[0].plot(x, x**2, "*", label='quadratic')  # Plot more data on the axes...
ax[0].plot(x, x**3,"x", label='cubic')  # ... and some more.
ax[0].set_xlabel('x label')  # Add an x-label to the axes.
ax[0].set_ylabel('y label')  # Add a y-label to the axes.
ax[0].set_title("Simple Plot")  # Add a title to the axes.
ax[0].legend()  # Add a legend.
ax[1].scatter(x, x, label='linear')  # scatter some data on the axes.
ax[1].scatter(x, x**2, label='quadratic')  # scatter more data on the axes...
ax[1].scatter(x, x**3, label='cubic')  # ... and some more.
ax[1].set_xlabel('x label')  # Add an x-label to the axes.
ax[1].set_ylabel('y label')  # Add a y-label to the axes.
ax[1].set_title("Simple Plot")  # Add a title to the axes.
ax[1].legend()  # Add a legend.
plt.show() # or : fig.show()

# plot画散点图。

# 可以添加颜色属性：color='b' /
# 'b'	蓝色
# 'g'	绿色
# 'r'	红色
# 'c'	青色
# 'm'	品红色
# 'y'	黄色
# 'k'	黑色
# 'w'	白色

# 画柱状图(ax.bar((x, x, label='linear',align='center'))，

# 以上三点可以参考: https://www.runoob.com/numpy/numpy-matplotlib.html