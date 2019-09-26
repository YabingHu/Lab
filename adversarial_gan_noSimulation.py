# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 13:41:35 2019

@author: yabinghu
"""
#tensorboard --logdir logs/gradient_tape
import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import json
import datetime
import argparse
import os
import pickle
parser = argparse.ArgumentParser(description='gannifiy')

parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50000)
parser.add_argument('--noise_dim', type=int, default=100)
parser.add_argument('--gen_num', type=int, default=5)
parser.add_argument('--sample_size', type=int, default=128)

args = parser.parse_args()

EPOCHS = args.epochs
noise_dim = args.noise_dim
gen_num = args.gen_num
BATCH_SIZE = args.batch_size

#with open('parameters/train_data') as json_file:
with open('parameters/train_data') as json_file:
    train_data =  np.asarray(json.load(json_file))
ones=np.ones([10000,2,1000])
train_data=train_data-ones

#random_vector_for_generation = tf.random_normal([num_examples_to_generate,noise_dim])
random_vector_for_generation =np.random.normal(0, 0.1, [gen_num, noise_dim])
test_noise=tf.convert_to_tensor(random_vector_for_generation, dtype=tf.float32)
with open('Results/random_vector_for_generation', 'wb') as write_file:
    pickle.dump(random_vector_for_generation,write_file)        

#%%
def make_generator_model():
    model = tf.keras.Sequential() 
    model.add(tf.keras.layers.Dense(1*500*10, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())

    model.add(tf.keras.layers.Reshape((1, 500, 10)))
    assert model.output_shape == (None, 1, 500, 10)
    
    model.add(tf.keras.layers.Conv2DTranspose(5, (2, 2), strides=(2, 2), padding='same', use_bias=False))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    assert model.output_shape == (None, 2, 1000, 5)  
    
    model.add(tf.keras.layers.Conv2DTranspose(1, (2, 2), strides=(1, 1), padding='same', activation='tanh',use_bias=False))
    assert model.output_shape == (None, 2, 1000, 1)  

    return model


#%%
def make_discriminator_model():
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(5, (2, 2), strides=(2, 2), padding='same', input_shape=[2, 1000, 1]))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
      
    model.add(tf.keras.layers.Conv2D(10, (2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dropout(0.3))
       
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1))
    #print(model.output_shape)
    return model
    

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


#%%

def generator_loss(discriminator_output):
    
    disc_loss = cross_entropy(tf.ones_like(discriminator_output), discriminator_output)

    return disc_loss

def discriminator_loss(pred_real, pred_gen):
    # [1,1,...,1] with real output since it is true and we want our generated examples to look like it
    real_loss = cross_entropy(tf.ones_like(pred_real), pred_real)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = cross_entropy(tf.zeros_like(pred_gen), pred_gen)

    total_loss = {'real_loss':real_loss, 'generated_loss': generated_loss}

    return total_loss

generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)


#%%
#data shape=(32, 2, 1000, 1)

def train_step(epoch,data,cnt):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])#(32,100)


    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generator_output = generator(noise, training=True)#shape=(32, 2, 1000, 1)
        pred_gen = discriminator(generator_output, training=True)#shape=(32, 1)
        pred_real = discriminator(data, training=True)


        print('finish one batch for epoch'+str(epoch)+' batch'+'_'+str(cnt))

        
        disc_accuracy = tf.reduce_mean(tf.to_float(pred_gen < 0))
        
        print('discriminator accuracy: {}'.format(disc_accuracy))
        
        gen_losses = generator_loss(pred_gen)
        gen_loss = gen_losses

        disc_losses = discriminator_loss(pred_real, pred_gen)
        disc_loss = tf.reduce_sum([ disc_losses[k] for k in disc_losses.keys()])

        
        # print losses
        print('generator loss: {}'.format(gen_loss))
        print('discriminator loss: {}'.format(disc_loss))
        
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))
    
    return gen_loss, disc_loss,disc_accuracy
#train_step = tf.contrib.eager.defun(train_step)


#%%
def train(epoch,train_dataset):
    cnt=0
    for data in train_dataset:
        cnt+=1
        gen_loss,disc_loss,disc_accuracy=train_step(epoch,data,cnt)
        global_step.assign_add(1)
        with data_summary_writer.as_default(), tf.contrib.summary.always_record_summaries():  
             tf.contrib.summary.scalar('disc_loss', disc_loss)
             tf.contrib.summary.scalar('disc_accuracy', disc_accuracy)
             tf.contrib.summary.scalar('gen_loss', gen_loss)
    save_data(generator,discriminator,epoch,test_noise)
    

def save_data(generator, discriminator,epoch, test_input):
    generator_output = generator(test_input, training=False)
    predictions = discriminator(generator_output, training=False)
    if epoch %  100==0:
        with open('Results/a_b_'+str(epoch),'w') as write_file:
                json.dump(generator_output.numpy()[:,:,:,0].tolist(),write_file)
    if epoch>= 5000 and epoch % 1000==0:
        checkpoint.save(file_prefix = checkpoint_prefix)
    with data_summary_writer.as_default(), tf.contrib.summary.always_record_summaries():  
             tf.contrib.summary.scalar('noise_predictions', tf.to_float(predictions < 0))
    print('finish saving data!')


if __name__ == '__main__':
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    data_log_dir = 'logs/gradient_tape/' + current_time + '/data'
    data_summary_writer =tf.contrib.summary.create_file_writer(data_log_dir, flush_millis=10000)    
    global_step = tf.train.get_or_create_global_step()
    generator = make_generator_model()
    discriminator = make_discriminator_model()

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,discriminator_optimizer=discriminator_optimizer,generator=generator,discriminator=discriminator)
    
    for epoch in range(1,EPOCHS+1):
        print('start training epeoch',epoch)
        index=np.random.choice(train_data.shape[0], args.sample_size, replace=False)
        train_data_sub=train_data[index]
        train_data_sub = train_data_sub.reshape(train_data_sub.shape[0], 2, 1000, 1).astype('float32')
        BUFFER_SIZE = train_data_sub.shape[0]
        train_dataset = tf.data.Dataset.from_tensor_slices(train_data_sub).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
        train(epoch,train_dataset) 
        
    
            


#tensorboard --logdir logs/gradient_tape
