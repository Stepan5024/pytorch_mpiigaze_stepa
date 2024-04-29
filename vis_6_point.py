import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define the 6 points-based face model coordinates
points = [
    [-45.0967681, -21.3128582, 21.3128582, 45.0967681, -26.2995769, 26.2995769],
    [-0.483773045, 0.483773045, 0.483773045, -0.483773045, 68.5950353, 68.5950353],
    [2.39702984, -2.39702984, -2.39702984, 2.39702984, 0, 0]
]

# Unpack points into x, y, z coordinates for plotting
x = points[0]
y = points[1]
z = points[2]

# Create a new figure for 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the points for eye corners
ax.scatter(x[:4], y[:4], z[:4], c='blue')

# Plot the points for mouth corners
ax.scatter(x[4:], y[4:], z[4:], c='red')

# Connect the eye corners to form the eye outlines
eye_corners = list(range(4)) + [0]
ax.plot(np.array(x)[eye_corners], np.array(y)[eye_corners], np.array(z)[eye_corners], color='blue')

# Connect the mouth corners
ax.plot(np.array(x)[4:], np.array(y)[4:], np.array(z)[4:], color='red')

# Label the axes
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')

# Add title
ax.set_title('6 Points-Based Face Model Visualization')

# Show the plot
plt.show()
