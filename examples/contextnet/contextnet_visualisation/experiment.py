import pickle
import os
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from tensorflow_asr.fullGrad import fullgrad
from plotly.offline import iplot

number_of_points = 9
small_range = -1.0
large_range = 1.0

xcoordinates = np.linspace(small_range, large_range, num=number_of_points)
ycoordinates = np.linspace(small_range, large_range, num=number_of_points)

xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)

directory = '/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet/contextnet_visualisation/checkpoints_cn_lists'
parent_directory = '/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet'

figure_directory = "figs_wave_model_6june"


def create_viz(loss_list, acc_list, figure_directory, filename, title="none"):
    print(filename)
    # define global size 16 for text and fonttsize, legends, title etc
    # plot the loss functions
    # experiment with cmap
    # clear that level concept in images
    plt.figure(figsize=(5, 5))
    if title != "none":
        plt.title("Contour 2D")
    CS = plt.contour(xcoord_mesh, ycoord_mesh, loss_list, 20, zorder=1, cmap='terrain', linestyles='--')
    plt.clabel(CS, inline=1, fontsize=8)
    plt.xticks(fontsize=8, fontname="Courier New")
    plt.yticks(fontsize=8, fontname="Courier New")
    plt.savefig(figure_directory + "/original_contour/" + filename + "OriginalContour.png")

    plt.figure(figsize=(5, 5))
    if title != "none":
        plt.title("Contour 2D ")
    plt.contour(xcoord_mesh, ycoord_mesh, loss_list, 20, zorder=1, cmap='terrain', linestyles='--')
    CS = plt.contourf(xcoord_mesh, ycoord_mesh, loss_list, 20, zorder=1, cmap='terrain', linestyles='--')
    plt.clabel(CS, fontsize=8, inline=1, fmt='%2.1f')
    plt.xticks(fontsize=8, fontname="Courier New")
    plt.yticks(fontsize=8, fontname="Courier New")
    plt.colorbar(CS)
    plt.savefig(figure_directory + "/original_contour_color/" + filename + "OriginalContourColor.png")

    plt.figure(figsize=(5, 5))
    if title != "none":
        plt.title("Contour 2D Log Scale")
    CS = plt.contour(xcoord_mesh, ycoord_mesh, np.log(loss_list), 20, zorder=1, cmap='terrain', linestyles='--')
    plt.clabel(CS, inline=1, fontsize=8)
    plt.xticks(fontsize=8, fontname="Courier New")
    plt.yticks(fontsize=8, fontname="Courier New")
    plt.savefig(figure_directory + "/log_contour/" + filename + "LogScale.png")

    plt.figure(figsize=(5, 5))
    if title != "none":
        plt.title("Contour 2D Log Scale Color")
    plt.contour(xcoord_mesh, ycoord_mesh, np.log(loss_list), 20, zorder=1, cmap='terrain', linestyles='--')
    CS = plt.contourf(xcoord_mesh, ycoord_mesh, np.log(loss_list), 20, zorder=1, cmap='terrain', linestyles='--')
    plt.clabel(CS, fontsize=8, inline=1, fmt='%2.1f')
    plt.xticks(fontsize=8, fontname="Courier New")
    plt.yticks(fontsize=8, fontname="Courier New")
    plt.savefig(figure_directory + "/log_contour_color/" + filename + "LogScale.png")

    data = [
        go.Surface(
            x=xcoord_mesh, y=ycoord_mesh,
            z=(loss_list.max() - loss_list.min()) * (acc_list - acc_list.min()) / (acc_list.max() - acc_list.min() + 1e-8) + loss_list.min(),
            showscale=False, opacity=0.6, colorscale='Cividis',
        ),
        go.Surface(
            x=xcoord_mesh, y=ycoord_mesh, z=loss_list, colorscale='Jet', cmin=0, cmax=30000, opacity=0.9,
            contours=go.surface.Contours(z=go.surface.contours.Z(show=True, usecolormap=True, project=dict(z=True), ),
                                         )
        )
    ]

    layout = go.Layout(autosize=False,
                       scene=dict(dict(
                           xaxis=dict(range=[-1, 1],
                                      backgroundcolor="rgb(163, 221, 240)",
                                      gridcolor="white",
                                      showbackground=True,
                                      zerolinecolor="white", tick0=-1.5, dtick=0.5, title_font_family="Courier New"),
                           yaxis=dict(range=[-1, 1],
                                      backgroundcolor="rgb(91, 122, 133)",
                                      gridcolor="white",
                                      showbackground=True,
                                      zerolinecolor="white", tick0=-1.5, dtick=0.5, title_font_family="Courier New"),
                           zaxis=dict(range=[1, 12],
                                      backgroundcolor="rgb(204, 231, 240)",
                                      gridcolor="white",
                                      showbackground=True,
                                      zerolinecolor="white", tick0=2, dtick=1, title_font_family="Courier New")),
                           camera=dict(eye=dict(x=2, y=2, z=1.5))),
                       margin=dict(l=50, r=10, b=20, t=10),
                       width=500, height=500)
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        font_family="Courier New")
    fig.update_xaxes(title_font_family="Courier New")
    fig.update_yaxes(title_font_family="Courier New")

    iplot(fig)

    fig.write_image(figure_directory + "/loss_accuracy/" + filename + "Loss_Accuracy.png", scale=1)


    data = [
        go.Surface(
            x=xcoord_mesh, y=ycoord_mesh,
            z=(np.log(loss_list).max() - np.log(loss_list).min()) * (acc_list - acc_list.min()) / (acc_list.max() - acc_list.min() + 1e-8) + np.log(
                loss_list).min(),
            showscale=False, opacity=0.6, colorscale='Cividis',
        ),
        go.Surface(
            x=xcoord_mesh, y=ycoord_mesh, z=np.log(loss_list), colorscale='Jet', cmin=0, cmax=12, opacity=0.9,
            contours=go.surface.Contours(z=go.surface.contours.Z(show=True, usecolormap=True, project=dict(z=True), ),
                                         )
        )
    ]
    layout = go.Layout(autosize=False,
                       scene=dict(dict(
                           xaxis=dict(range=[-1, 1],
                                      backgroundcolor="rgb(163, 221, 240)",
                                      gridcolor="white",
                                      showbackground=True,
                                      zerolinecolor="white", tick0=-1.5, dtick=0.5, title_font_family="Courier New" ),
                           yaxis=dict(range=[-1, 1],
                                      backgroundcolor="rgb(91, 122, 133)",
                                      gridcolor="white",
                                      showbackground=True,
                                      zerolinecolor="white", tick0=-1.5, dtick=0.5, title_font_family="Courier New"),
                           zaxis=dict(range=[1, 12],
                                      backgroundcolor="rgb(204, 231, 240)",
                                      gridcolor="white",
                                      showbackground=True,
                                      zerolinecolor="white", tick0=2, dtick=1, title_font_family="Courier New")),
                           camera=dict(eye=dict(x=2, y=2, z=1.5))),
                       margin=dict(l=50, r=10, b=20, t=10),
                       width=500, height=500)
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        font_family="Courier New")
    fig.update_xaxes(title_font_family="Courier New")
    fig.update_yaxes(title_font_family="Courier New")
    iplot(fig)
    fig.write_image(figure_directory + "/log_loss_accuracy/" + filename + "Log_Loss_Accuracy.png", scale=1)


def make_directories(figure_directory, parent_directory):
    os.mkdir(figure_directory)
    os.chdir(figure_directory)
    os.mkdir("log_contour")
    os.mkdir("log_contour_color")
    os.mkdir("log_loss_accuracy")
    os.mkdir("loss_accuracy")
    os.mkdir("original_contour")
    os.mkdir("original_contour_color")
    os.chdir(parent_directory)
    print(os.getcwd())

try:
    make_directories(figure_directory, parent_directory)
except Exception as e:
    print("exception occured while creating the directory")
    print("Beware dude, trying to overwrite images")
    print(e)

# m = "/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet/before_fit_contextnet.h5"

for file in sorted(os.listdir(directory)):
    filename = os.path.join(directory, file)
    print("filenmes ", "/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet/model_5th_iter_trained_01.pkl")
    with open(filename, "rb") as f:
        x_temp = pickle.load(f)

    loss_list = x_temp['loss_list'][0]
    acc_list_greedy_char = x_temp['greedy_char'][0]
    acc_list_greedy_wer = x_temp['greedy_wer'][0]
    acc_list_beam_wer = x_temp['beam_wer'][0]
    acc_list_beam_char = x_temp['beam_char'][0]
    acc_list = acc_list_beam_char

    create_viz(loss_list, acc_list_beam_char, figure_directory, "model_wave_lrcn")
    break

# Do the 2d level curve plots in log space too
