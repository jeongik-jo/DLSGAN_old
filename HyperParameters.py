import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
#tf.config.experimental.enable_tensor_float_32_execution(False)
import tensorflow.keras as kr

mapper_optimizer = kr.optimizers.Adam(learning_rate=0.001 * 0.01, beta_1=0.0, beta_2=0.99)
decoder_optimizer = kr.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99)
discriminator_optimizer = kr.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99)
lr_decay = 0.02

inception_model = tf.keras.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)


image_resolution = 128

latent_vector_dim = 512


gp_weight = 10.0
dis_enc_weight = 1.0
gen_enc_weight = 1.0
is_dls_gan = True
var_vector_size = 512


batch_size = 16
save_image_size = 8

train_data_size = -1
test_data_size = -1
shuffle_test_dataset = True
epochs = 50

load_model = False

evaluate_model = True
fid_batch_size = batch_size
epoch_per_evaluate = 1

is_latent_normal = True

if is_latent_normal:
    def latent_dist_func(shape):
        return tf.random.normal(shape)
    def latent_entropy_func(latent_scale_vector):
        return tf.reduce_sum(tf.math.log(latent_scale_vector * tf.sqrt(2.0 * 3.141592 * tf.exp(1.0))))
    def latent_add_noises(latent_vectors, noise_size):
        noise = tf.random.normal([1, noise_size, latent_vector_dim], stddev=0.3)
        noised_vectors_sets = latent_vectors[:, tf.newaxis, :] + noise
        return noised_vectors_sets

    latent_interpolation_value = 2.0
else:
    def latent_dist_func(shape):
        return tf.random.uniform(shape, minval=-tf.sqrt(3.0), maxval=tf.sqrt(3.0))
    def latent_entropy_func(latent_scale_vector):
        return tf.reduce_sum(tf.math.log(latent_scale_vector * 2 * tf.sqrt(3.0)))
    def latent_add_noises(latent_vectors, noise_size):
        noise = tf.random.normal([1, noise_size, latent_vector_dim], stddev=0.3)
        noised_vectors_sets = tf.clip_by_value(latent_vectors[:, tf.newaxis, :] + noise,
                                               clip_value_min=-tf.sqrt(3.0), clip_value_max=tf.sqrt(3.0))
        return noised_vectors_sets

    latent_interpolation_value = tf.sqrt(3.0)
