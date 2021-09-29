import tensorflow as tf
import tensorflow.keras as kr
import HyperParameters as hp
import Dataset


@tf.function
def _train_step(mapper: kr.Model, decoder: kr.Model, discriminator: kr.Model, real_images: tf.Tensor, var_vectors):
    real_images = Dataset.resize_and_normalize(real_images)

    with tf.GradientTape(persistent=True) as tape:
        batch_size = real_images.shape[0]
        if hp.is_dls_gan:
            latent_scale_vector = tf.sqrt(tf.reduce_mean(var_vectors, axis=0, keepdims=True))
            latent_scale_vector = tf.sqrt(tf.cast(hp.latent_vector_dim, dtype='float32')) * latent_scale_vector / tf.norm(latent_scale_vector, axis=-1, keepdims=True)
            latent_vectors = hp.latent_dist_func([batch_size, hp.latent_vector_dim])

            fake_images = decoder(mapper(latent_vectors * latent_scale_vector, training=True), training=True)
            fake_adv_values, rec_latent_vectors = discriminator(fake_images, training=True)
            enc_losses = tf.reduce_mean(tf.square(rec_latent_vectors - latent_vectors) * tf.square(latent_scale_vector), axis=-1)
        else:
            latent_vectors = hp.latent_dist_func([batch_size, hp.latent_vector_dim])
            fake_images = decoder(mapper(latent_vectors, training=True), training=True)
            fake_adv_values, rec_latent_vectors = discriminator(fake_images, training=True)
            enc_losses = tf.reduce_mean(tf.square(rec_latent_vectors - latent_vectors), axis=-1)
        var_vectors = tf.concat([var_vectors[1:], tf.reduce_mean(tf.square(rec_latent_vectors), axis=0, keepdims=True)], axis=0)

        with tf.GradientTape() as inner_tape:
            inner_tape.watch(real_images)
            real_adv_values, _ = discriminator(real_images, training=True)
        real_gradients = inner_tape.gradient(real_adv_values, real_images)
        gp_losses = tf.reduce_sum(tf.square(real_gradients), axis=[1, 2, 3])

        discriminator_losses = tf.squeeze(tf.nn.softplus(-real_adv_values) + tf.nn.softplus(fake_adv_values)) \
                             + hp.dis_enc_weight * enc_losses + hp.gp_weight * gp_losses
        generator_losses = tf.squeeze(tf.nn.softplus(-fake_adv_values)) + hp.gen_enc_weight * enc_losses

    hp.mapper_optimizer.apply_gradients(
        zip(tape.gradient(generator_losses, mapper.trainable_variables),
            mapper.trainable_variables)
    )

    hp.decoder_optimizer.apply_gradients(
        zip(tape.gradient(generator_losses, decoder.trainable_variables),
            decoder.trainable_variables)
    )

    hp.discriminator_optimizer.apply_gradients(
        zip(tape.gradient(discriminator_losses, discriminator.trainable_variables),
            discriminator.trainable_variables)
    )

    del tape

    return tf.reduce_mean(enc_losses), var_vectors


def train(mapper: kr.Model, decoder: kr.Model, discriminator: kr.Model, dataset, var_vectors):
    enc_losses = []
    for data in dataset:
        enc_loss, var_vectors = _train_step(mapper, decoder, discriminator, data, var_vectors)
        enc_losses.append(enc_loss)
    mean_enc_loss = tf.reduce_mean(enc_losses)

    return mean_enc_loss, var_vectors
