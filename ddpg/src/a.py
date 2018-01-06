#!/usr/bin/env python


import numpy as np
# import tensorflow as tf
# import pickle
# import matplotlib.pyplot as plt

# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation


# def update_line(num, data, line):
#     line.set_data(data[..., :num])
#     return line,

# fig1 = plt.figure()

# data = np.random.rand(2, 25)
# plt.plot([1, 2, 3, 4], [2,4,6,8])
# plt.title('test')



# plt.show()


# output = open('/home/mtb/rbt_ws/src/ddpg/src/data.txt', 'wb')
# b = 5
# for i in range(4):
#     b +=5
#     c=[i,b]
#     pickle.dump(c, output)

# output.close()
# output = open('/media/mtb/W10X64_PROV/ddpg_ros/src/rewards.txt', 'rb')
# a = list()
# x= list()
# y=list()
# while True:
#     try:
#         a .append(pickle.load(output))
#         x.append(a[-1][0])
#         y.append(a[-1][1])
      
#     except:
#        break

# output.close()
# print x
# print y
# plt.plot(x,y)
# plt.show()

my_array = np.array([[1., 3., 5., 7., 9.],
                   [-2., 0., 2., 4., 6.],
                   [-6., -3., 0., 3., 6.]])

x_vals = np.array([my_array, my_array + 1])

print x_vals    