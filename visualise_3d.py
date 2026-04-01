import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

csv_path = 'results/car_3d_trajectory.csv'
print(f"Loading trajectory data from: {csv_path}")

if not os.path.exists(csv_path):
    print("Error: Could not find the CSV file!")
    exit()

df = pd.read_csv(csv_path)

# 2. 3D Scatter Plot
print("Generating interactive 3D plot...")
fig = px.scatter_3d(
    df, 
    x='World_X', 
    y='World_Y', 
    z='World_Z',
    color=df.index,
    color_continuous_scale='turbo',
    title='Interactive 3D Car Trajectory',
    labels={'color': 'Frame Sequence'}
)

# Formatting the lines and markers
fig.update_traces(
    marker=dict(size=4, opacity=0.8),
    line=dict(width=2, color='DarkSlateGrey'),
    selector=dict(mode='markers')
)
fig.data[0].update(mode='markers+lines')

#  Lock the Aspect Ratio
fig.update_layout(
    scene=dict(
        xaxis_title='World X',
        yaxis_title='World Y',
        zaxis_title='World Z (Altitude)',
        aspectmode='data' # Forces 1 unit of X to equal 1 unit of Y and Z
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

html_output = 'results/interactive_3d_trajectory.html'
fig.write_html(html_output)
print(f"Saved interactive 3D map to: {html_output}")

fig.show()
