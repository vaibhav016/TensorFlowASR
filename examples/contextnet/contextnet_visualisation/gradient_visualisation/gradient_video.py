import glob
import os

import cv2

img_array = []
size = (10, 10)


def make_directories():
    current_working_directory_abs = os.getcwd()
    gradient_directory_abs = os.path.join(current_working_directory_abs, "video")
    try:
        os.mkdir(gradient_directory_abs)
    except Exception as e:
        print("--------------gradients plots directory already exists-----------------")
        print("--------------The contents will be over-ridden-------------------")
        return gradient_directory_abs

    return gradient_directory_abs

directory_to_save = make_directories()

fname1 = os.path.join(os.getcwd(), "gradient2_plots") + "/*.png"
# fname1 = '/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet/contextnet_visualisation/gradient_visualisation/grad_vis_4/*.png'

for filename1 in (sorted(glob.glob(fname1))):
    print(filename1)
    image1 = cv2.imread(filename1)
    height, width, layers = image1.shape
    size = (width, height)
    print(size)
    img_array.append(image1)

out = cv2.VideoWriter(directory_to_save + "/gradient_vis.avi", cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
