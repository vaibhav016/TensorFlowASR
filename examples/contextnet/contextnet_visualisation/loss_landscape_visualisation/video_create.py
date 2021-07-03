import glob
import os

import cv2

def make_directory():
    current_working_directory_abs = os.getcwd()
    video_directory_abs = os.path.join(current_working_directory_abs, "video")
    try:
        os.mkdir(video_directory_abs)
    except Exception as e:
        print("--------------video directory already exists-----------------")
        print("--------------The contents will be over-ridden-------------------")
        return video_directory_abs

    return video_directory_abs

def create_video(directory, project_type):
    img_array = []
    size = (10,10)
    for filename in sorted(glob.glob(directory)):
        img = cv2.imread(filename)
        print(filename)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter("projectaaaaaa_log_loss_accuracy_" + project_type + ".avi", cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


img_array = []
size = (10,10)
figures_working_dir = os.path.join(os.getcwd(), "figs")

fname1 = figures_working_dir+'/log_loss_accuracy/*.png'
fname2 = figures_working_dir+'/log_contour/*.png'

video_directory = make_directory()


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

    img_array.append(vis)

filename = video_directory + "/contour_video.avi"
print(filename)
out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()
