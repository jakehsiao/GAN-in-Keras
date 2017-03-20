# %matplotlib inline
import os,random
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import theano as th
import theano.tensor as T
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, Deconv2D, UpSampling2D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
import matplotlib.pyplot as plt
import seaborn as sns
import random, sys, keras
import pickle as cPickle
import cv2
import glob
from keras.models import Model
from IPython import display
from datetime import datetime

sys.path.append("../common")
from keras.utils import np_utils
from tqdm import tqdm


'''generator'''

def generative_model(nch, shp):
    # Build Generative model ...
    g_input = Input(shape=[100])
    H = Dense(nch*((shp[0]//2)**2), init='glorot_normal')(g_input) # change this
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Reshape( [shp[0]//2, shp[0]//2, nch] )(H) # change this
    H = UpSampling2D(size=(2, 2))(H)
    H = Convolution2D(nch//2, 3, 3, border_mode='same', init='glorot_uniform')(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Convolution2D(nch//4, 3, 3, border_mode='same', init='glorot_uniform')(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = Convolution2D(shp[2], 1, 1, border_mode='same', init='glorot_uniform')(H)
    g_V = Activation('sigmoid')(H)
    generator = Model(g_input,g_V)
    generator.compile(loss='binary_crossentropy', optimizer=opt)
    generator.summary()
    return generator

def discriminative_model(shp, dropout_rate=0.25):
    # Build Discriminative model ...
    d_input = Input(shape=shp)
    H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(d_input)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode = 'same', activation='relu')(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    H = Flatten()(H)
    H = Dense(256)(H)
    H = LeakyReLU(0.2)(H)
    H = Dropout(dropout_rate)(H)
    d_V = Dense(2,activation='softmax')(H)
    discriminator = Model(d_input,d_V)
    discriminator.compile(loss='categorical_crossentropy', optimizer=dopt)
    discriminator.summary()
    return discriminator

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val
        
def gan(nch, shp, dropout_rate=0.25):
    # Freeze weights in the discriminator for stacked training
    generator = generative_model(nch, shp)
    discriminator = discriminative_model(shp, dropout_rate)
    make_trainable(discriminator, False)
    # Build stacked GAN model
    gan_input = Input(shape=[100])
    H = generator(gan_input)
    gan_V = discriminator(H)
    GAN = Model(gan_input, gan_V)
    GAN.compile(loss='categorical_crossentropy', optimizer=opt)
    GAN.summary()
    return generator, discriminator, GAN

def plot_loss(losses,time_now="0"):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.figure(figsize=(10,8))
    plt.plot(losses["d"], label='discriminitive loss')
    plt.plot(losses["g"], label='generative loss')
    plt.legend()
    #plt.show()
    plt.savefig("loss_plot_%s.png"%time_now)
    
def plot_gen(generator, n_ex=16,dim=(4,4), figsize=(10,10) ):
    noise = np.random.uniform(0,1,size=[n_ex,100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,:,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

'''train discriminator'''
def train_discriminator(generator, discriminator, XT, epoch=5, batch_size=32):
    # Pre-train the discriminator network ...
    noise_gen = np.random.uniform(0,1,size=[XT.shape[0],100])
    generated_images = generator.predict(noise_gen)
    X = np.concatenate((XT, generated_images))
    n = XT.shape[0]
    y = np.zeros([2*n,2])
    y[:n,1] = 1
    y[n:,0] = 1

    make_trainable(discriminator,True)
    discriminator.fit(X,y, nb_epoch=epoch, batch_size=batch_size)

def train_for_n(generator, discriminator, GAN, nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):
    global losses, X_train
    for e in tqdm(range(nb_epoch)):  
        
        # Make generative images
        image_batch = X_train[np.random.randint(0,X_train.shape[0],size=BATCH_SIZE),:,:,:]    
        noise_gen = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        generated_images = generator.predict(noise_gen)
        
        # Train discriminator on generated images
        # CHANGE THIS TO change the 0,1 distribution
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1
        
        make_trainable(discriminator,True)
        d_loss  = discriminator.train_on_batch(X,y)
        losses["d"].append(d_loss)
    
        # train Generator-Discriminator stack on input noise to non-generated output class
        noise_tr = np.random.uniform(0,1,size=[BATCH_SIZE,100])
        y2 = np.zeros([BATCH_SIZE,2])
        y2[:,1] = 1
        
        make_trainable(discriminator,False)
        g_loss = GAN.train_on_batch(noise_tr, y2 )
        losses["g"].append(g_loss)
        
        # Updates plots
        if e%plt_frq==plt_frq-1:
            #plot_loss(losses)
            #plot_gen(generator)
            evaluate_gen_and_save(generator,e)
            time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            generator.save("Generative_%s_%d.h5"%(time_now,e))

def evaluate_generator(generator):
    noise_gen = np.random.uniform(0,1,size=[100,100])
    generated_images = generator.predict(noise_gen)
    
    evaluate_img = np.zeros([720,720,3])
    for i in range(10):
        for j in range(10):
            evaluate_img[72*i:72*(i+1),72*j:72*(j+1),:]=generated_images[i*10+j]
    
    return evaluate_img
            
def evaluate_gen_and_save(generator,e):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    eval_dir = "evaluate_gen"
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)
    evaluate_img = evaluate_generator(generator)
    plt.imsave("%s/iter_%s_%d.png"%(eval_dir,time_now,e), evaluate_img)
                          
    
    
    
    
if __name__=="__main__":
    print("Version 1.1, updated 320")
    print("load the dataset")
    X_train = np.load("GAN_DS_YCrCb_km6.npy")
    '''get the shape and opt'''

    shp = X_train.shape[1:]
    print(shp)

    dropout_rate = 0.25
    # Optim

    opt = Adam(lr=1e-4)
    dopt = Adam(lr=1e-5)
    nch = 512 # TUNE
    
    G, D, GAN = gan(nch, shp)
    print("==="*3)
    print("start training the discriminator")
    train_discriminator(G, D, X_train, epoch=3) # TUNE
    # set up loss storage vector
    losses = {"d":[], "g":[]}
    print("start GAN training")
    train_for_n(G, D, GAN, nb_epoch=150, plt_frq=50,BATCH_SIZE=128) # TUNE
    
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    G.save("Generative_%s.h5"%time_now)
    D.save("Discriminative_%s.h5"%time_now)
    #plot_loss(losses,time_now)
