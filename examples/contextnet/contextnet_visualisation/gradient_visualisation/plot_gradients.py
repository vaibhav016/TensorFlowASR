import os
import pickle

import matplotlib
import tensorflow as tf

matplotlib.use('Agg') # No pictures displayed
import librosa.display

import matplotlib.pyplot as plt

def make_directories():
    current_working_directory_abs = os.getcwd()
    gradient_directory_abs = os.path.join(current_working_directory_abs, "gradient_plots")
    try:
        os.mkdir(gradient_directory_abs)
    except Exception as e:
        print("--------------gradients plots directory already exists-----------------")
        print("--------------The contents will be over-ridden-------------------")
        return gradient_directory_abs

    return gradient_directory_abs


gradient_directory = os.path.join(os.getcwd(), "grad_list_lrcn")
directory_to_save = make_directories()

for index, file in enumerate(sorted(os.listdir(gradient_directory))):
    filename = os.path.join(gradient_directory, file)
    print("filenmes ",filename)
    with open(filename, "rb") as f:
        x_temp = pickle.load(f)

    name = file.split('.')[0].split('_')[-1]

    images_check = x_temp['input_image']
    gradients_check = x_temp["integrated_gradients"]
    random_gradients = x_temp["random_integrated_gradients"]

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(25, 14), facecolor=(1, 1, 1))
    fig.suptitle('LRCN checkpoint ' + str(index), fontsize=16)
    for i, j in enumerate(zip(images_check, gradients_check)):
        print(images_check[i+1].shape)
        print(gradients_check[i].shape)

        ax[0][0].set(title=" Log Mel spectrogram for male voice")
        img = librosa.display.specshow(images_check[i].numpy().T, sr=16000,
                                       fmax=8000, x_axis='time', y_axis='mel', ax=ax[0][0], alpha=1)
        plt.colorbar(img, ax=ax[0][0])
        ax[0][0].label_outer()


        ax[0][1].set(title=" Integrated Gradient Attribution ")
        # normalized_grads = tf.linalg.normalize(gradients_check[i])
        # S_dB = librosa.power_to_db(gradients_check[i].numpy().T, ref=np.max)
        img2 = librosa.display.specshow(tf.math.square(tf.abs(gradients_check[i])).numpy().T, sr=16000,
                                        fmax=8000, x_axis='time', y_axis='mel',cmap=plt.get_cmap("Reds") ,ax=ax[0][1],  alpha=1)
        # img2.set_clim(vmin=-0.5, vmax=0.5)
        plt.colorbar(img2, ax=ax[0][1])
        ax[0][1].label_outer()

        # normalized_grads_2 = tf.linalg.normalize(random_gradients[i])
        ax[0][2].set(title=" Random Node Integrated Gradient Attribution ")
        # S_dB = librosa.power_to_db(random_gradients[i].numpy().T, ref=np.max)
        img2 = librosa.display.specshow(tf.math.square(tf.abs(random_gradients[i])).numpy().T, sr=16000,
                                        fmax=8000, x_axis='time', y_axis='mel', cmap=plt.get_cmap("Reds") ,ax=ax[0][2], alpha=1)
        # img2.set_clim(vmin=-0.5, vmax=0.5)
        plt.colorbar(img2, ax=ax[0][2])
        ax[0][2].label_outer()

############################################################  2nd row of female#######################################################

        ax[1][0].set(title=" Log Mel spectrogram for female voice")
        img = librosa.display.specshow(images_check[i + 1].numpy().T, sr=16000,
                                       fmax=8000, x_axis='time', y_axis='mel', ax=ax[1][0], alpha=1)
        plt.colorbar(img, ax=ax[1][0])
        ax[1][0].label_outer()

        # normalized_grads = tf.linalg.normalize(gradients_check[i+1])
        # S_dB = librosa.power_to_db(gradients_check[i+1].numpy().T, ref=np.max)
        img2 = librosa.display.specshow(tf.math.square(tf.abs(gradients_check[i+1])).numpy().T, sr=16000,
                                        fmax=8000, x_axis='time', y_axis='mel', cmap=plt.get_cmap("Reds") ,ax=ax[1][1],  alpha=1)
        # img2.set_clim(vmin=-0.5, vmax=0.5)
        plt.colorbar(img2, ax=ax[1][1])
        ax[1][1].label_outer()

        # normalized_grads = tf.linalg.normalize(random_gradients[i + 1])

        # S_dB = librosa.power_to_db(random_gradients[i+1].numpy().T, ref=np.max)
        img2 = librosa.display.specshow(tf.math.square(tf.abs(random_gradients[i+1])).numpy().T, sr=16000,
                                        fmax=8000, x_axis='time', y_axis='mel', cmap=plt.get_cmap("Reds") , ax=ax[1][2], alpha=1)
        # img2.set_clim(vmin=-0.5, vmax=0.5)
        plt.colorbar(img2, ax=ax[1][2])
        ax[1][2].label_outer()


        # this break means that only 2 images(male and female) will be displayed in a plot.
        # This is not a sanity break. Its purposeful.
        break


    plt.savefig(directory_to_save + "Grad" + name +".png")



