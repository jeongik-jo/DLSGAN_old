import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import tensorflow as tf
#tf.config.experimental.enable_tensor_float_32_execution(False)
import tensorflow.keras as kr

mapper_optimizer = kr.optimizers.Adam(learning_rate=0.001 * 0.01, beta_1=0.0, beta_2=0.99) # optimizer for mapper of generator
decoder_optimizer = kr.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99) # optimizer for decoder of generator
discriminator_optimizer = kr.optimizers.Adam(learning_rate=0.001, beta_1=0.0, beta_2=0.99) # optimizer for discriminator (and encoder)
lr_decay = 0.02 # learning rate decay per epoch

inception_model = tf.keras.applications.InceptionV3(weights='imagenet', pooling='avg', include_top=False)


image_resolution = 128

latent_vector_dim = 512


gp_weight = 10.0 # r1 loss weight
dis_enc_weight = 1.0
gen_enc_weight = 1.0
is_dls_gan = True # if false, use MSE
var_vector_size = 512 # DLSGAN traces past "var_vector_size * batch_size" samples for variance vector


batch_size = 16
save_image_size = 8 # save_image_size^2 is number of samples for saved images

train_data_size = -1 # train data size. If -1, use all samples
test_data_size = -1 # test data size. If -1, use all samples
shuffle_test_dataset = True
epochs = 50

load_model = False # if True, use saved model in ./models. if False, create new model.

evaluate_model = True
fid_batch_size = batch_size # batch size for calculate FID
epoch_per_evaluate = 1

is_latent_normal = True # if True, Z~N(0, 1). if False, z~U(-sqrt(3), sqrt(3))

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
