# The script loads the training data and shows the first image.
# This may be helpful for visualizing the data and show how to load the data using scipy

import scipy as sp
import matplotlib.pyplot as plt
train = sp.genfromtxt('mnist_train.txt', delimiter=',')
img = train[0, 1:].astype(sp.uint8).reshape(28, 28)
plt.gray()
plt.imshow(img)
plt.show()
