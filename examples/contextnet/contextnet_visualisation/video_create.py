import cv2
import numpy as np
import glob

def create_video(directory, project_type):
    img_array = []
    size = 10
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
size = 10
fname1 = '/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet/figs_1june_cn/log_loss_accuracy/*.png'
fname2 = '/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet/figs_1june_cn/log_contour/*.png'


for filename1, filename2 in zip(sorted(glob.glob(fname2)), sorted(glob.glob(fname1))):
    print(filename1)
    print(filename2)

    image1 = cv2.imread(filename1)
    image2 = cv2.imread(filename2)
    height, width, layers = image1.shape
    sizee = (width, height)
    print(sizee)
    height, width, layers = image2.shape
    sizee = (width, height)
    print(sizee)
    # image1 = cv2.resize(image1, (500, 500))
    # height, width, layers = image1.shape
    # sizee = (width, height)
    # print(sizee)

    vis = cv2.hconcat([image1, image2])
    height, width, layers = vis.shape
    size = (width, height)

    # vis = np.concatenate((image1, image2), axis=0)
    img_array.append(vis)


out = cv2.VideoWriter("loss_contour_3d_cn.avi", cv2.VideoWriter_fourcc(*'DIVX'), 1, size)

for i in range(len(img_array)):
    out.write(img_array[i])

out.release()

# import matplotlib.pyplot as plt
# plt.imshow(vis)
# plt.show()