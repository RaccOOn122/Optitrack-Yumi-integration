import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load the CSV file
data = pd.read_csv('ProcessedSortedClap/processed_sorted_clap_05.csv', delimiter=';')

# data = pd.read_csv('ProcessedSortedWave/processed_sorted_wave_01.csv', delimiter=';')


# Extract unique rigid body names
rigid_bodies = data['Rigid Body Name'].unique()

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Calculate axis limits to center (0, 0, 0)
x_min, x_max = data['Relative X'].min(), data['Relative X'].max()
y_min, y_max = data['Relative Y'].min(), data['Relative Y'].max()
z_min, z_max = data['Relative Z'].min(), data['Relative Z'].max()

max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
mid_x = (x_max + x_min) / 2.0
mid_y = (y_max + y_min) / 2.0
mid_z = (z_max + z_min) / 2.0

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Initialize lines for each rigid body
lines = {}
for body in rigid_bodies:
    lines[body], = ax.plot([], [], [], label=body)

# Label the axes
ax.set_xlabel('Position X')
ax.set_ylabel('Position Y')
ax.set_zlabel('Position Z')
ax.set_title('3D Model Location Over Time (Live Animation)')

# Add a legend
ax.legend()

# Function to initialize the plot
def init():
    for line in lines.values():
        line.set_data([], [])
        line.set_3d_properties([])
    return lines.values()

# Function to update the plot
def update(frame):
    for body in rigid_bodies:
        body_data = data[data['Rigid Body Name'] == body]
        body_data = body_data[body_data['Frame Number'] <= frame]
        
        lines[body].set_data(body_data['Relative X'], body_data['Relative Y'])
        lines[body].set_3d_properties(body_data['Relative Z'])
    return lines.values()

# Get the range of frames
frames = data['Frame Number'].unique()

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=10)

# Show the animation
plt.show()
