import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.metrics import pairwise_distances_argmin

bs = np.loadtxt("/home/sberton2/Scaricati/_boresights_LOLA_ch12345_day_laser2_fov.inc")
print(bs)
# plt.scatter(x=bs[:, 0], y=bs[:, 1])
# plt.show()

ax = plt.subplot()
ax.scatter(x=bs[::1, 0], y=bs[::1, 1])

X = bs[:, :2].copy(order='C')
db = DBSCAN(eps=0.00009, min_samples=10).fit(X)
labels = db.labels_
print(labels)
# exit()

# ax.scatter(x=mbk_means_cluster_centers[:, 0], y=mbk_means_cluster_centers[:, 1])

cm = ['r', 'k', 'g', 'y', 'c', 'r', 'k', 'g', 'y', 'c']
for idx in range(10):
    ax.scatter(x=bs[np.where(labels == idx), 0],
               y=bs[np.where(labels == idx), 1],
               c=cm[idx])

plt.show()

# replace all channels by channel 4
ch4in = bs[np.where(labels == 3)]
ch4out = bs[np.where(labels == 8)]
print(ch4in)

for i in [0, 1, 2, 4]:
    bs[np.where(labels == i)] = ch4in

for i in [5, 6, 7, 9]:
    bs[np.where(labels == i)] = ch4out

print(bs)
np.savetxt('/home/sberton2/Scaricati/_boresights_LOLA_ch44444_day_laser2_fov.inc', bs,
           fmt='%.12e', delimiter='  ')
