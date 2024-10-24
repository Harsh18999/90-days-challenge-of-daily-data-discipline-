from sklearn.datasets import make_blobs
import plotly.graph_objects as go
import numpy as np
import streamlit as st

class Point:
  def __init__(self,cordinates,position,neighbor_points,cluster):
    self.cordinates = cordinates
    self.position = position
    self.neighbor_points = neighbor_points
    self.cluster = cluster


class DBSCAN:

  def __init__(self,eps,min_points):
    self.eps = eps
    self.min_points = min_points

  def _assign_position(self,X,point):
    neighbor_points = []
    point_pos = 0

    for i in range(len(X)):
      dist = 0
      for j in range(len(X[i])):
        dist += (X[i][j] - point[j])**2
      dist = np.sqrt(dist)

      if dist <= self.eps:
        neighbor_points.append(i)

    if len(neighbor_points) >= self.min_points:
      point_pos = 1
    elif len(neighbor_points) > 1:
      point_pos = 2
    else: point_pos = 3

    return neighbor_points,point_pos

  def fit(self,X):
    current_cluster = 0
    points = []
    frames = [ ]
    colors = [-2]*len(X)

    for i,point in enumerate(X):
      Neighbor_Points, Point_Pos = self._assign_position(X,point)
      points.append(Point(point,Point_Pos,Neighbor_Points,1-Point_Pos))
      colors[i] = 1 -Point_Pos
      frames.append(self._create_frame(X,colors,'Process - Selecting Core Points',i))

    for i in range(len(X)):
      if points[i].cluster == 0 :
        current_cluster += 1
        points[i].cluster = current_cluster
        colors[i] = current_cluster
        frames.append(self._create_frame(X,colors,f'Process - Assign Cluster {current_cluster}',i))
        self._find_cluster_points(X,current_cluster,points,i,colors,frames)

    self.plot_animation(frames,X)
    return 

  def _find_cluster_points(self,X, current_cluster, points, i,colors,frames):

    cluster_members = points[i].neighbor_points
    j = 0
    while j < len(cluster_members):
      expention_point = cluster_members[j]
      if points[expention_point].cluster == -1:
        colors[expention_point] = current_cluster
        frames.append(self._create_frame(X,colors,f'Process - Assign Cluster {current_cluster}',expention_point))
        points[expention_point].cluster = current_cluster

      elif points[expention_point].cluster == 0:
        colors[expention_point] = current_cluster
        frames.append(self._create_frame(X,colors,f'Process - Assign Cluster {current_cluster}',expention_point))
        points[expention_point].cluster = current_cluster
        cluster_members += points[expention_point].neighbor_points
      j += 1

  def _create_frame(self,X,colors,process,index):
    # Scatter plot for data points colored by cluster
    trace_data_points = go.Scatter(
        x=X[:, 0], y=X[:, 1],
        mode='markers',
        marker=dict(color=colors, size=10, colorscale='Viridis', line=dict(width=0.2)),
        name=process
    )
    trace_point = go.Scatter(
            x= [X[index][0]], y= [X[index][1]],
            mode='markers',
            marker=dict(color='red', size=10, symbol='x', line=dict(width=0.2)),
            name='Processing Point'
        ) 
    # Return frame with current data points and centroids
    return go.Frame(data=[trace_data_points, trace_point], layout=go.Layout(title_text=process)) 
  
  def plot_animation(self, frames, X):
        # Create a figure for the animation
        fig = go.Figure(
            data=[frames[0].data[0], frames[0].data[1]],  # Initial data points and centroids
            layout=go.Layout(
                title="DBSCAN Algorithm Animation",
                xaxis=dict(range=[X[:, 0].min() - 1, X[:, 0].max() + 1]),
                yaxis=dict(range=[X[:, 1].min() - 1, X[:, 1].max() + 1]),
                updatemenus=[{
                    "buttons": [{
                        "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                        "label": "Play",
                        "method": "animate"
                    }],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }]
            ),
            frames=frames
        )

        # Display the plot in Streamlit
        st.plotly_chart(fig)

def Animation(n_samples,n_cluster,cluster_std):
  X, y = make_blobs(n_samples=n_samples, centers=n_cluster, n_features=2, random_state=42,cluster_std=cluster_std)
  Model = DBSCAN(2,5)
  Model.fit(X)

Animation(100,3,2)