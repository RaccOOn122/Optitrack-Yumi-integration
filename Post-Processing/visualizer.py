import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Load the CSV file
data = pd.read_csv('Training/Ultimate-Test/Samples/true/clap/received_data_20250131_162120.csv', delimiter=';')  # Update with actual file path

# Ensure required columns exist
expected_columns = {'Right_X', 'Right_Y', 'Right_Z', 'Left_X', 'Left_Y', 'Left_Z'}
missing_columns = expected_columns - set(data.columns)
if missing_columns:
    raise ValueError(f"Missing columns in CSV file: {missing_columns}")

# Swap Y (Depth) and Z (Height)
data['Right_Y'], data['Right_Z'] = data['Right_Z'], data['Right_Y']
data['Left_Y'], data['Left_Z'] = data['Left_Z'], data['Left_Y']

# Set up the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Adjust the camera view for better orientation
ax.view_init(elev=20, azim=-90)  # Adjust elevation and azimuth for better depth/height perception

# Calculate axis limits to center (0, 0, 0)
x_min, x_max = data[['Right_X', 'Left_X']].min().min(), data[['Right_X', 'Left_X']].max().max()
y_min, y_max = data[['Right_Z', 'Left_Z']].min().min(), data[['Right_Z', 'Left_Z']].max().max()  # Now Y represents height
z_min, z_max = data[['Right_Y', 'Left_Y']].min().min(), data[['Right_Y', 'Left_Y']].max().max()  # Now Z represents depth

max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
mid_x = (x_max + x_min) / 2.0
mid_y = (y_max + y_min) / 2.0
mid_z = (z_max + z_min) / 2.0

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Initialize lines for left and right hand
lines = {
    'Right Hand': ax.plot([], [], [], label='Right Hand')[0],
    'Left Hand': ax.plot([], [], [], label='Left Hand')[0]
}

# Label the axes (Adjusted for Correct Y-Z Swap)
ax.set_xlabel('Position X')
ax.set_ylabel('Position Y (Height)')  # Y now represents height
ax.set_zlabel('Position Z (Depth)')  # Z now represents depth
ax.set_title('3D Hand Motion Over Time (Live Animation)')

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
    frame_data = data.iloc[:frame]
    lines['Right Hand'].set_data(frame_data['Right_X'], frame_data['Right_Z'])  # Z is now depth
    lines['Right Hand'].set_3d_properties(frame_data['Right_Y'])  # Y is now height
    lines['Left Hand'].set_data(frame_data['Left_X'], frame_data['Left_Z'])  # Z is now depth
    lines['Left Hand'].set_3d_properties(frame_data['Left_Y'])  # Y is now height
    return lines.values()

# Get the range of frames
frames = range(len(data))

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=10)

# Show the animation
plt.show()
