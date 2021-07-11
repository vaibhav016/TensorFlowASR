import glob
import os

import cv2
from tqdm import tqdm


def make_directory(loss_lists_directory):
    current_working_directory_abs = loss_lists_directory
    video_directory_abs = os.path.join(current_working_directory_abs, "video")
    try:
        os.mkdir(video_directory_abs)
    except Exception as e:
        print("--------------video directory already exists-----------------")
        print("--------------The contents will be over-ridden-------------------")
        return video_directory_abs

    return video_directory_abs

loss_check_list_directory = "/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet/contextnet_visualisation/loss_landscape_visualisation/check_lists"

for i, dir in enumerate(tqdm(sorted(os.listdir(loss_check_list_directory)))):
    if dir == ".DS_Store":
        continue

    loss_lists_directory = os.path.join(loss_check_list_directory, dir)
    video_directory = make_directory(loss_lists_directory)


    img_array = []
    size = (10,10)

    figures_working_dir = os.path.join(loss_lists_directory, "figs")

    fname1 = figures_working_dir+'/log_loss_accuracy/*.png'
    fname2 = figures_working_dir+'/log_contour/*.png'

    index = 1
    for filename1, filename2 in zip(sorted(glob.glob(fname2)), sorted(glob.glob(fname1))):
        print(filename1)
        print(filename2)

        image1 = cv2.imread(filename1)
        image2 = cv2.imread(filename2)
        height, width, layers = image1.shape
        size = (width, height)
        print(size)
        height, width, layers = image2.shape
        size = (width, height)
        print(size)

        vis = cv2.hconcat([image1, image2])
        height, width, layers = vis.shape
        size = (width, height)

        font = cv2.QT_FONT_NORMAL
        # org
        org = (460, 28)

        # fontScale
        fontScale = 0.8

        # Blue color in BGR
        color = (0, 0, 0)

        # Line thickness of 2 px
        thickness = 1

        # Using cv2.putText() method
        image = cv2.putText(vis, 'Epoch ' + str(index), org, font,
                            fontScale, color, thickness, cv2.LINE_4)

        img_array.append(image)
        index = index + 1

    filename = video_directory + "/log_loss.mp4"
    print(filename)
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 1, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()
