import math
import os
import pickle

from tensorflow_asr.utils import env_util
env_util.setup_environment()

DEFAULT_YAML = "/Users/vaibhavsingh/Desktop/TensorFlowASR/examples/contextnet/configs_local/config_macbook.yml"


def make_directories():
    current_working_directory_abs = os.getcwd()
    gradient_directory_abs = os.path.join(current_working_directory_abs, "gradient_lists")
    try:
        os.mkdir(gradient_directory_abs)
    except Exception as e:
        print("--------------gradientlist directory already exists-----------------")
        print("--------------The contents will be over-ridden-------------------")
        return gradient_directory_abs

    return gradient_directory_abs


directory_to_save = make_directories()

from tensorflow_asr.configs.config import Config
from tensorflow_asr.datasets.asr_dataset import ASRSliceDataset
from tensorflow_asr.featurizers.speech_featurizers import TFSpeechFeaturizer
from tensorflow_asr.featurizers.text_featurizers import CharFeaturizer
from tensorflow_asr.models.transducer.contextnet import ContextNet
from tensorflow_asr.optimizers.schedules import TransformerSchedule
from tensorflow_asr.utils import env_util

env_util.setup_environment()
import tensorflow as tf

tf.keras.backend.clear_session()
tf.config.optimizer.set_experimental_options({"auto_mixed_precision": False})
strategy = env_util.setup_strategy([0])

config = Config(DEFAULT_YAML)
model_directory = config.learning_config.running_config.checkpoint_directory
last_trained_model = os.path.join(model_directory, sorted(os.listdir(model_directory))[-1])

speech_featurizer = TFSpeechFeaturizer(config.speech_config)

text_featurizer = CharFeaturizer(config.decoder_config)
tf.random.set_seed(0)

visualisation_dataset = ASRSliceDataset(
    speech_featurizer=speech_featurizer,
    text_featurizer=text_featurizer,
    **vars(config.learning_config.gradient_dataset_vis_config)
)

batch_size = 1
visualisation_gradient_loader = visualisation_dataset.create(batch_size)

contextnet = ContextNet(**config.model_config, vocabulary_size=text_featurizer.num_classes)
contextnet.make(speech_featurizer.shape)

contextnet.load_weights(last_trained_model, by_name=True)
contextnet.add_featurizers(speech_featurizer, text_featurizer)

optimizer = tf.keras.optimizers.Adam(
    TransformerSchedule(
        d_model=contextnet.dmodel,
        warmup_steps=config.learning_config.optimizer_config.pop("warmup_steps", 10000),
        max_lr=(0.05 / math.sqrt(contextnet.dmodel))
    ),
    **config.learning_config.optimizer_config
)

contextnet.compile(
    optimizer=optimizer,
    steps_per_execution=1,
    global_batch_size=1,
    blank=text_featurizer.blank
)
encoder = contextnet.layers[0]

activated_node_list = []
random_activated_node_list = []

for i, j in visualisation_gradient_loader:
    inputs = tf.Variable(i["inputs"])
    inputs_length = tf.Variable(i["inputs_length"])
    signal = tf.Variable(i["signal"])

    encoder_output = encoder.call_feature_output([inputs, inputs_length, signal])
    activated_channels = tf.norm(encoder_output, axis=1)
    activated_node_index = tf.math.argmax(activated_channels, axis=1).numpy()

    activated_node_list.append(activated_node_index[0])
    random_activated_node_list.append(3)


@tf.function
def get_integrated_gradients(encoder, mel_spec, inputs_length, signal, activated_node_index, random_node_index):
    m_steps = 50
    baseline = tf.zeros(shape=mel_spec.shape)
    alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps + 1)
    print("alphas", alphas.shape)
    alphas_x = alphas[:, tf.newaxis, tf.newaxis]
    print("alphas_x", alphas_x.shape)
    baseline_x = tf.expand_dims(baseline, axis=0)
    print("baseline ", baseline_x.shape)
    input_x = tf.expand_dims(mel_spec, axis=0)
    print("input", input_x.shape)
    delta = input_x - baseline_x
    interpolated_images = baseline_x + alphas_x * delta
    print("final images", interpolated_images.shape)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(interpolated_images)
        images = tf.expand_dims(interpolated_images, axis=-1)
        print(images.shape)
        encoder_output = encoder.call_feature_output([images, inputs_length, signal])
        gradients = tape.gradient(encoder_output[:, :, activated_node_index], interpolated_images)

        random_gradients = tape.gradient(encoder_output[:, :, random_node_index], interpolated_images)

    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
    random_grads = (random_gradients[:-1] + random_gradients[1:]) / tf.constant(2.0)

    integrated_gradients = tf.math.reduce_mean(grads, axis=0)
    integrated_random_gradients = tf.math.reduce_mean(random_grads, axis=0)

    return integrated_gradients, integrated_random_gradients


for filename in os.listdir(model_directory):
    if not filename.endswith(".h5"):
        print(filename)
        continue

    gradient_file = filename.split('.')[0]
    model_name = os.path.join(model_directory, filename)
    print(model_name)

    contextnet.load_weights(model_name, by_name=True)
    encoder = contextnet.layers[0]

    m = 0
    images_check = []
    gradients_check = []
    random_gradients_check = []
    for i, j in visualisation_gradient_loader:
        inputs = tf.Variable(i["inputs"])
        inputs_length = tf.Variable(i["inputs_length"])
        signal = tf.Variable(i["signal"])

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(inputs)
            encoder_output = encoder.call_feature_output([inputs, inputs_length, signal])
            gradients = tape.gradient(encoder_output[:, :, activated_node_list[m]], inputs)
            random_gradients = tape.gradient(encoder_output[:, :, random_activated_node_list[m]], inputs)

        interated_gradients, random_integrated_gradients = get_integrated_gradients(encoder, tf.squeeze(inputs),
                                                                                    inputs_length, signal,
                                                                                    activated_node_list[m],
                                                                                    random_activated_node_list[m])

        gradients_check.append(interated_gradients)
        random_gradients_check.append(random_integrated_gradients)

        images_check.append(tf.squeeze(inputs))

        print("integrated_gradients shape=========", interated_gradients.shape, random_integrated_gradients.shape)
        m = m + 1

    dd = {'input_image': images_check,
          'integrated_gradients': gradients_check,
          'random_integrated_gradients': random_gradients_check
          }

    with open(directory_to_save + filename + ".pkl", 'wb') as f:
        pickle.dump(dd, f)
