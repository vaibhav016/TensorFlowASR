import pickle

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from plotly.offline import iplot

with open("loss_&_acc.pkl","rb") as f:
    x_temp = pickle.load(f)

loss_list = x_temp['loss_list'][0]
acc_list_greedy_char = x_temp['greedy_char'][0]
acc_list_greedy_wer = x_temp['greedy_wer'][0]
acc_list_beam_wer = x_temp['beam_wer'][0]
acc_list_beam_char = x_temp['beam_char'][0]

number_of_points = 9
small_range = -1.0
large_range = 1.0

xcoordinates = np.linspace(small_range, large_range, num=number_of_points)
ycoordinates = np.linspace(small_range, large_range, num=number_of_points)

xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)

def create_viz(loss_list, acc_list, title="none"):
    # define global size 16 for text and fonttsize, legends, title etc
    # plot the loss functions
    plt.figure(figsize=(18, 6))
    plt.title("Original Contour")
    CS = plt.contour(xcoord_mesh, ycoord_mesh, loss_list, 10, zorder=1, cmap='terrain', linestyles='--')
    plt.clabel(CS, inline=1, fontsize=16)
    plt.savefig("figs/Original Contour.png")

    plt.figure(figsize=(18, 6))
    plt.title("Original Contour with Color")
    plt.contour(xcoord_mesh, ycoord_mesh, loss_list, 10, zorder=1, cmap='terrain', linestyles='--')
    CS = plt.contourf(xcoord_mesh, ycoord_mesh, loss_list, 10, zorder=1, cmap='terrain', linestyles='--')
    plt.clabel(CS, fontsize=16, inline=0, fmt='%2.1f')
    plt.colorbar(CS)
    plt.savefig("figs/Original Contour with Color.png")

    plt.figure(figsize=(18, 6))
    plt.title("Log Scale")
    CS = plt.contour(xcoord_mesh, ycoord_mesh, np.log(loss_list + 1e-8), 10, zorder=1, cmap='terrain', linestyles='--');
    plt.clabel(CS, fontsize=8, inline=1)
    plt.savefig("figs/Log Scale.png")


    data = [
        go.Surface(
            x=xcoord_mesh, y=ycoord_mesh,
            z=(loss_list.max() - loss_list.min()) * (acc_list - acc_list.min()) / (acc_list.max() - acc_list.min() + 1e-8) + loss_list.min(),
            showscale=False, opacity=0.6, colorscale='Cividis',
        ),
        go.Surface(
            x=xcoord_mesh, y=ycoord_mesh, z=loss_list, colorscale='Jet', opacity=0.9,
            contours=go.surface.Contours(z=go.surface.contours.Z(show=True, usecolormap=True, project=dict(z=True), ),
                                         )
        )
    ]
    layout = go.Layout(title='Loss / Accuracy', autosize=True, scene=dict(camera=dict(eye=dict(x=1.87, y=0.88, z=-0.64))),
                       margin=dict(l=65, r=50, b=65, t=90))
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    fig.write_image("figs/Loss_Accuracy.png")

    data = [
        go.Surface(
            x=xcoord_mesh, y=ycoord_mesh,
            z=(np.log(loss_list).max() - np.log(loss_list).min()) * (acc_list - acc_list.min()) / (acc_list.max() - acc_list.min() + 1e-8) + np.log(
                loss_list).min(),
            showscale=False, opacity=0.6, colorscale='Cividis',
        ),
        go.Surface(
            x=xcoord_mesh, y=ycoord_mesh, z=np.log(loss_list), colorscale='Jet', opacity=0.9,
            contours=go.surface.Contours(z=go.surface.contours.Z(show=True, usecolormap=True, project=dict(z=True), ),
                                         )
        )
    ]
    layout = go.Layout(title='Log Scale Loss / Accuracy', autosize=True, scene=dict(camera=dict(eye=dict(x=1.87, y=0.88, z=-0.64))),
                       margin=dict(l=65, r=50, b=65, t=90))
    fig = go.Figure(data=data, layout=layout)
    iplot(fig)
    fig.write_image("figs/Log_Loss_Accuracy.png")

create_viz(loss_list,acc_list_beam_char)


