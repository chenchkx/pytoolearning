# from: https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py
# https://blog.csdn.net/weixin_38753213/article/details/105885338

# figure 中各组件叫什么可以参考：
# https://matplotlib.org/tutorials/introductory/usage.html#sphx-glr-tutorials-introductory-usage-py

# figure 相当于一张画布 一张用于绘画的白纸， aex是在画布上待绘画的区域,相关解释可以参考：
# https://blog.csdn.net/weixin_38753213/article/details/105885338
#

import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1,ncols=1)     # Create a figure containing a single axes.
ax.plot([1, 2, 3, 4], [1, 4, 2, 3], label='fold line')         # Plot some data on the axes.
ax.legend()    # Add a legend(图例，说明，解释). The label of data
plt.show()

# 也可以使用下面这样的形式，但是推荐上面的（逻辑清晰）.
plt.plot([1, 2, 3, 4], [1, 4, 2, 3],label='fold line')
plt.legend()
plt.show()