import glob
import os

import cv2

img_array = []
size = (10, 10)

fname1 = os.path.join(os.getcwd(), "grad_list") + "/*.png"
# fname1 = '/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet/contextnet_visualisation/gradient_visualisation/grad_vis_4/*.png'

for filename1 in (sorted(glob.glob(fname1))):
    print(filename1)
    image1 = cv2.imread(filename1)
    height, width, layers = image1.shape
    size = (width, height)
    print(size)
    img_array.append(image1)

file_directory = os.path.join(os.getcwd(), "video")
out = cv2.VideoWriter(file_directory + "gradient_vis.avi", cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
