import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 1, 1, 1, 1, 1],
              [1, 0, 0, 0, 1, 1],
              [1, 0, 1, 0, 0, 1],
              [1, 0, 1, 1, 0, 1],
              [1, 0, 0, 0, 0, 1],
              [1, 1, 1, 1, 1, 1]], dtype='float') * 0.6

fig, ax = plt.subplots()
ax.axis('off')
ax.imshow(X, cmap='Greens', vmin=0, vmax=1)
# plt.matshow(X, cmap='Greens', vmin=0, vmax=1)
plt.scatter(1, 1, s=800, marker='>')

# Save the figure with not margin
plt.savefig("grid_plot.pdf", format="pdf", bbox_inches="tight", pad_inches=0)

# plt.show()
