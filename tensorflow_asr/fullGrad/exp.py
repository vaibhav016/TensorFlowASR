import ast

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras import backend as K
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.models import *
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRTFRecordDataset, ASRSliceDataset
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer, SentencePieceFeaturizer, CharFeaturizer
from tensorflow_asr.models.transducer.contextnet import ContextNet
from tensorflow_asr.utils import app_util


config_path = "/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet/configs_local/config_macbook_sanity.yml"
config = Config(config_path)
speech_featurizer = TFSpeechFeaturizer(config.speech_config)
text_featurizer = CharFeaturizer(config.decoder_config)


contextnet = ContextNet(**config.model_config, vocabulary_size=text_featurizer.num_classes)
contextnet.make(speech_featurizer.shape)

contextnet.load_weights("model_trained_01.h5", by_name=True)

# contextnet.summary(line_length=100)
contextnet.add_featurizers(speech_featurizer, text_featurizer)



from tensorflow_asr.fullGrad.fullgrad import FullGrad

with open('imagenet1000_clsidx_to_labels.txt') as imagenet_classes_file:
    imagenet_classes_dict = ast.literal_eval(imagenet_classes_file.read())

K.clear_session()
tf.compat.v1.disable_eager_execution()
base_models = [VGG16(weights='imagenet')]
base_preprocess = [vgg_preprocess]
base_model_names = ['VGG']

m1 = contextnet.layers[0].layers

for base_model in base_models:
    '''
    completeness check with random input eg: a tensor of ones
    '''
    input_ = np.ones(
        shape=(1, base_model.layers[0].input_shape[0][1], base_model.layers[0].input_shape[0][2], base_model.layers[0].input_shape[0][3])).astype(
        np.float32)
    print(input_.shape)
    '''
    since the fullgrad model deals with representation of relu layers, 
    last layer softmax is removed and it's input features are mul with 
    weights and added with bias of last layer.

    This process can be ignored as it is done to manually check 
    base model output and fullgrad sum. 
    
    '''
    newmodel = Model(base_model.input, base_model.layers[-2].output)
    feat2 = base_model.predict(tr)
    feat = newmodel.predict(input_)
    base_weights = base_model.get_weights()
    out_ = feat.dot(base_weights[-2]) + base_weights[-1]
    '''
    create fullgrad model and check for it's completeness
    Note: default num classes is 1000
    '''
    fullgrad = FullGrad(base_model)
    fullgrad.checkCompleteness(input_)
    print('###############################')

img_path = '/Users/vaibhavsingh/Desktop/TensorFlowASR/tensorflow_asr/fullGrad/images/2007_000256.jpeg'

for base_model, preprocess, name in zip(base_models, base_preprocess, base_model_names):
    '''
    image loading and preprocessing
    '''

    img = load_img(img_path, target_size=(base_model.layers[0].input_shape[0][1], base_model.layers[0].input_shape[0][2]))
    img = img_to_array(img)
    input_ = np.expand_dims(img, axis=0)
    input_ = preprocess(input_)

    '''
    since the fullgrad model deals with representation of relu layers, 
    last layer softmax is removed and it's input features are mul with 
    weights and added with bias of last layer.

    This process can be ignored as it is done to manually check 
    base model output and fullgrad sum.

    '''
    # newmodel=Model(base_model.input,base_model.layers[-2].output)
    # feat=newmodel.predict(input_)
    # base_weights=base_model.get_weights()
    # out_=feat.dot(base_weights[-2])+base_weights[-1]

    '''
    create fullgrad model and fullgrad saliency for the
    given input . fullgrad completeness check can be done
    or ignored here if prevously done once.

    Note: default num classes is 1000 , for custom model use
    FullGrad(base_model,num_classes)
    '''
    fullgrad = FullGrad(base_model, class_names=imagenet_classes_dict)
    # fullgrad.checkCompleteness(input_)
    cam = fullgrad.saliency(input_)
    cam = fullgrad.postprocess_saliency_map(cam[0])
    plt.title(name)
    plt.imshow(cv2.resize(plt.imread(img_path), (base_model.layers[0].input_shape[0][1], base_model.layers[0].input_shape[0][2])))
    plt.imshow(cam, alpha=0.5)
    plt.show()
