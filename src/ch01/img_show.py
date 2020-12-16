# coding: utf-8
import matplotlib.pyplot as plt
from matplotlib.image import imread

#  parameter 0 fix PNG header issue
img = imread('../dataset/sample.png', 0)
plt.imshow(img)

plt.show()
