import HyperParameters as hp
import Train
import time
import Models
import Dataset
import Evaluate
import datetime
import numpy as np
import os
import tensorflow as tf


def train_gan():
    generator, discriminator = Models.Generator(), Models.Discriminator()

    if hp.load_model:
        generator.load(), discriminator.load()

    train_dataset, test_dataset = Dataset.load_celeb_dataset()

    fids = []
    real_psnrs = []
    real_ssims = []
    fake_psnrs = []
    fake_ssims = []
    enc_losses = []
    latent_entropys = []

    for epoch in range(hp.epochs):
        print(datetime.datetime.now())
        print('epoch', epoch)
        start = time.time()
        enc_loss, var_vectors = Train.train(generator.mapper, generator.decoder, discriminator.discriminator, train_dataset, generator.var_vectors)

        hp.mapper_optimizer.lr = hp.mapper_optimizer.lr * (1.0 - hp.lr_decay)
        hp.decoder_optimizer.lr = hp.decoder_optimizer.lr * (1.0 - hp.lr_decay)
        hp.discriminator_optimizer.lr = hp.discriminator_optimizer.lr * (1.0 - hp.lr_decay)
        generator.var_vectors = var_vectors

        if hp.is_dls_gan:
            latent_scale_vector = tf.sqrt(tf.reduce_mean(var_vectors, axis=0, keepdims=True))
            latent_scale_vector = tf.sqrt(tf.cast(hp.latent_vector_dim, dtype='float32')) * latent_scale_vector / tf.norm(latent_scale_vector, axis=-1, keepdims=True)
        else:
            latent_scale_vector = tf.ones([1, hp.latent_vector_dim])
        print('enc loss:', enc_loss.numpy(), '\n')

        print('saving...')
        generator.save()
        discriminator.save()
        generator.save_images(test_dataset, discriminator.discriminator, latent_scale_vector, epoch)
        print('saved')
        print('time: ', time.time() - start, '\n')
        if hp.evaluate_model:
            if (epoch + 1) % hp.epoch_per_evaluate == 0:
                start = time.time()
                enc_losses.append(enc_loss)

                print('evaluating...', '\n')
                fid = Evaluate.get_fid(generator.mapper, generator.decoder, test_dataset, latent_scale_vector)
                fids.append(fid)
                print('fid:', fid.numpy())
                latent_entropy = hp.latent_entropy_func(latent_scale_vector)
                latent_entropys.append(latent_entropy)
                print('latent entropy:', latent_entropy.numpy())

                fake_psnr, fake_ssim = Evaluate.evaluate_fake(generator.mapper, generator.decoder, discriminator.discriminator, test_dataset, latent_scale_vector)
                fake_psnrs.append(fake_psnr), fake_ssims.append(fake_ssim)
                print('\nfake psnr:', fake_psnr.numpy(), '\nfake ssim:', fake_ssim.numpy())

                real_psnr, real_ssim = Evaluate.evaluate_real(generator.mapper, generator.decoder, discriminator.discriminator, test_dataset, latent_scale_vector)
                real_psnrs.append(real_psnr), real_ssims.append(real_ssim)
                print('\nreal psnr:', real_psnr.numpy(), '\nreal ssim:', real_ssim.numpy())

                print('\n', 'evaluated')
                print('time: ', time.time() - start, '\n')
                if not os.path.exists('./results'):
                    os.makedirs('./results')
                np.savetxt('./results/fids.txt', np.array(fids), fmt='%f')
                np.savetxt('./results/fake_psnrs.txt', np.array(fake_psnrs), fmt='%f')
                np.savetxt('./results/fake_ssims.txt', np.array(fake_ssims), fmt='%f')
                np.savetxt('./results/real_psnrs.txt', np.array(real_psnrs), fmt='%f')
                np.savetxt('./results/real_ssims.txt', np.array(real_ssims), fmt='%f')
                np.savetxt('./results/enc_losses.txt', np.array(enc_losses), fmt='%f')
                np.savetxt('./results/latent_entropys.txt', np.array(latent_entropys), fmt='%f')


train_gan()

