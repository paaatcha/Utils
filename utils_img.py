# -*- coding: utf-8 -*-
'''
This file has some auxiliary functions to load and handle images

Author: André Pacheco
Email: pacheco.comp@gmail.com

If you find some bug, please email-me
'''
from __future__ import print_function
from __future__ import division
from random import shuffle
import glob
import os
import tensorflow as tf
import numpy as np
from utils import one_hot_encoding


'''     
    This function returns a 2 list: a list of folders' name in a root folder and a list of
    all images' path in all folders
    For example: if we have the following tree:
        IMGs
            - A
                img1.png
                img2.png
            - B
                img3.ong
                img4.png
    The root folder is IMGs, its children will be A, B. paths will return a list composed by 
    IMGs/{children}/img{number}.png and fold_names = ['A', 'B']
    
    Input:
        path: root folder path
        img_ext: the image extension
        shuf: if you'd like to shuffle the list of paths set it as True
        scalar_feat_ext: the extension of a file of scalar feature
    Output:
        paths: a list of images' paths in all folders
        fold_names: a list of name of all folders' in the root folder    
'''
def get_path_from_folders (path, scalar_feat_ext=None, img_ext='jpg', shuf=True):
    paths = list()      
    fold_names = [nf for nf in os.listdir(path) if os.path.isdir(os.path.join(path, nf))]  
    scalar_feat = list()
            
    if (len(fold_names) == 0):
        folders = glob.glob(path)
        fold_names = [path.split('/')[-1]]
    else:
        folders = glob.glob(path + '/*')
    
    for fold in folders:                   
        paths += (glob.glob(fold+'/*.'+img_ext))            
    
    if (shuf):        
        shuffle(paths)    
        
    if (scalar_feat_ext is not None):
        for p in paths:
            scalar_feat.append( np.loadtxt(p.split('.')[0]+'_feat.'+scalar_feat_ext) )
            #scalar_feat_paths.append(p.split('.')[0]+'_feat.'+scalar_feat)
    
    return paths, np.asarray(scalar_feat), fold_names 

'''    
    This gets a list of images' path and get all labels from the inner folder each image is inside.
    For example: aaa/bbb/ccc/img.png, the label will be ccc. Each path will have its own label

    Input:
        path: root folder path
        n_samples: number of samples that you wanna load from the path list
        img_ext: the image extension
        shuf: if you'd like to shuffle the list of paths set it as True
        one_hot: if you'd like the one hot encoding set it as True
        scalar_feat_ext: the extension of a file of scalar feature
    Output:
        paths: a list of images' paths in all folders
        labels: a list of the labels related with the paths
        dict_labels: a python dictionary relating path and label

'''
def get_path_and_labels_from_folders (path, scalar_feat_ext=None, n_samples=None, img_ext='jpg', shuf=True, one_hot=True):
    labels = list()
    
    # Getting all paths
    paths, scalar_feat_paths, folds = get_path_from_folders (path, scalar_feat_ext, img_ext, shuf)    
    dict_labels = dict()
    
    
    value = 0
    for f in folds:
        if (f not in dict_labels):
            dict_labels[f] = value
            value += 1
    
    if (n_samples is not None):
        paths = paths[0:n_samples]
    
    for p in paths:
        lab = p.split('/')[-2]
        labels.append(dict_labels[lab])
        
    if (one_hot):                
        labels = one_hot_encoding(labels)
        
    return paths, labels, scalar_feat_paths, dict_labels  


'''
    It gets an image path and returns a tensor with the image loaded and its related label
    
    Input:
        path: the image path
        label: the image label
        size: a tupla with the a new width and height. If you don't wanns chenge the image size
              set it as None
        channels: the image's depth  
        scalar_feat: if you're also loading scalar features with the images, you
        should use this parameter
        
    Output:
        img: the tensor with the loaded image
        label: the image label
'''

def load_img_as_tensor (path, label, size=(128,128), channels=3, scalar_feat=None):
    img = tf.read_file(path)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    img_decoded = tf.image.decode_jpeg(img, channels=channels)
        
    # This will convert to float values in [0, 1]
    img = tf.image.convert_image_dtype(img_decoded, tf.float32)
    
    # Resizing
    if (size is not None):
        img = tf.image.resize_images(img, size)

    if (scalar_feat is not None):
        return img, scalar_feat, label
    else:
        return img, label
    

'''
    It gets an tensor with an image loaded and runs some augmentation operations
    
    Input:
        image: the tensor with the loaded image
        label: the image label
        flip: set True to perform a flip augmentation
        bright: set True to perform a bright augmentation
        sat: set True to perform a saturation augmentation
        scalar_feat: if you're also loading scalar features with the images, you
        should use this parameter

'''
def get_aug_tf(image, label, flip=False, bright=False, sat=False, scalar_feat=None):
    if (flip):
        image = tf.image.random_flip_left_right(image)
        
    if (bright):
        image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
        
    if (sat):
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    if (scalar_feat is not None):
        return image, scalar_feat, label
    else:
        return image, label


'''
    It gets as parameter a list of paths and labels and returns the dataset according to tf.data.Dataset.
    
    Input:
        paths: the list of imagens path
        labels: the labels for each image in the path's list
        is_train: set True if this dataset is for training phase
        params: it's a python dictionary with the following keys:
            'img_size': a tuple containing width x height
            'channels': an integer representing the image's depth
            'augmentation': set True if you wanna perform the get_aug_tf. If it's False, you don't need to set
                            the parameters below.
                'flip': set True to perform a flip augmentation
                'bright': set True to perform a bright augmentation
                'sat': set it True to perform a saturation augmentation
            'shuffle': set it True if you wanna shuffle the dataset
            'repeat': set it True if you wanna repeat the dataset
            'threads': integer represeting the number of threads to processing the images' load
            'batch_size': an integer representing the batch size
        scalar_feat: if you're also loading scalar features with the images, you
        should use this parameter
            
    Output:
        inputs: a python dictionary containing get_next iterators for the image and labels, and the 
                make_initializable_iterator
        
        dataset: the tf.data.Dataset configured for the given data
'''
def get_dataset_tf(paths, labels, is_train, params, scalar_feat=None, verbose=True):
        
    if (scalar_feat is not None):
        get_aug = lambda x, s, y: get_aug_tf (x, y, params['flip'], params['bright'], params['sat'], s)    
        get_img = lambda x, s, y: load_img_as_tensor (x, y, params['img_size'], params['channels'], s)
    else:
        get_aug = lambda x, y: get_aug_tf (x, y, params['flip'], params['bright'], params['sat'])    
        get_img = lambda x, y: load_img_as_tensor (x, y, params['img_size'], params['channels'])
    
    if (verbose):   
        if (scalar_feat is not None):
            print ("\n******************\nLoading", len(paths), " images and", labels.shape, "scalar features", "With", labels.shape, " labels\n********************\n")        
        else:
            print ("\n******************\nLoading", len(paths), " images", "With", labels.shape, " labels\n********************\n")        
    
    if (is_train):
        if (scalar_feat is not None):
            dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(scalar_feat), tf.constant(labels)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(labels)))
            
        dataset = dataset.shuffle(len(paths))
        dataset = dataset.map(get_img, num_parallel_calls=params['threads'])
        
        
        if (params['augmentation']):
            dataset = dataset.map(get_aug, num_parallel_calls=params['threads'])
        
        if (params['repeat']):
            dataset = dataset.repeat()
        dataset = dataset.batch(params['batch_size'])
        dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
        
    else:
        if (scalar_feat is not None):
            dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(scalar_feat), tf.constant(labels)))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((tf.constant(paths), tf.constant(labels)))
            
        dataset = dataset.map(get_img, num_parallel_calls=params['threads'])
        
        if (params['repeat']):
            dataset = dataset.repeat()
            
        dataset = dataset.batch(params['batch_size'])
        dataset = dataset.prefetch(1)  # make sure you always have one batch ready to serve
        

    # Create reinitializable iterator from dataset
    iterator = dataset.make_initializable_iterator()
    iterator_init_op = iterator.initializer
    
    if (scalar_feat is not None):
        images, scalar_feat, labels = iterator.get_next()
        inputs = {'images': images, 'scalar_feat': scalar_feat, 'labels': labels, 'iterator_init_op': iterator_init_op}
    else:
        images, labels = iterator.get_next()
        inputs = {'images': images, 'labels': labels, 'iterator_init_op': iterator_init_op}
    
    return inputs, dataset

'''
    This function creates a folder tree to populate files in it
    
    Input:
        path: the root folder path
        folders: a list of strings representing the name of the folders will be created inside the path folder
        train_test_val: if you wann create TRAIN, TEST and VAL folders
'''
def create_dirs (path, folders=['A', 'B'], train_test_val=False):        
    
    # Checking if the folder already exists
    if (not os.path.isdir(path)):
        os.mkdir(path)
        
    if (train_test_val):
        if (not os.path.isdir(path + '/' + 'TEST')):
            os.mkdir(path + '/' + 'TEST')
        if (not os.path.isdir(path + '/' + 'TRAIN')):
            os.mkdir(path + '/' + 'TRAIN')                
        if (not os.path.isdir(path + '/' + 'VAL')):
            os.mkdir(path + '/' + 'VAL')
        
    for folder in folders:          
        if (train_test_val):
            if (not os.path.isdir(path + '/TRAIN/' + folder)):
                os.mkdir(path + '/TRAIN/' + folder)         
            if (not os.path.isdir(path + '/TEST/' + folder)):                                                   
                os.mkdir(path + '/TEST/' + folder)
            if (not os.path.isdir(path + '/VAL/' + folder)):
                os.mkdir(path + '/VAL/' + folder)
        else:               
            if (not os.path.isdir(path + '/' + folder)):
                os.mkdir(path + '/' + folder)


'''
    It gets as input a path tree without train, test and validation sets and returns a new folder tree with all sets.
    It's easier to explain with using an example (lol).
        Dataset:
            A:
                img...
            B: 
                img...
    It returns:
        Dataset:
            TRAIN:
                A:
                    imgs...
                B:
                    imgs...
            TEST:
                A:
                    imgs...
                B:
                    imgs...
            VAL:
                A:
                    imgs...
                B:
                    imgs...
    
    Input:
        path_in: the root folder that you wanna split in the train, test and val sets
        path_out: the root folder that will receive the new tree organization
        tr: a float meaning the % of images for the training set
        te: a float meaning the % of images for the test set
        tv: a float meaning the % of images for the validation set
        shuf: set it as True if you wanna shuffle the images
        verbose: set it as True to print information on the screen
    Outpur:
        The new folder tree with all images splited into train, test and val
    

'''
def split_folders_train_test_val (path_in, path_out, scalar_feat_ext=None, img_ext="jpg", tr=0.8, te=0.1, tv=0.1, shuf=False, verbose=False):
        
    if (tr+te+tv != 1.0):
        print ('tr, te and tv must sum up 1.0')
        raise ValueError
        
    folders = [nf for nf in os.listdir(path_in) if os.path.isdir(os.path.join(path_in, nf))]
    
    create_dirs (path_out, folders, True)   
    
    for lab in folders:            
        path_imgs = glob.glob(path_in + '/' + lab + '/*.'+img_ext)        
        
        if shuf:
            shuffle(path_imgs)
        
        N = len(path_imgs)
        n_test = int(round(te*N))
        n_val = int(round(tv*N))
        n_train = N - n_test - n_val        
        
        if (verbose):
            print ('Working on ', lab)
            print ('Total: ', N, ' | Train: ', n_train, ' | Test: ', n_test, ' | Val: ', n_val, '\n')
        
        path_test = path_imgs[0:n_test]
        path_val = path_imgs[n_test:(n_test+n_val)]
        path_train = path_imgs[(n_test+n_val):(n_test+n_val+n_train)]
        
        
        if (scalar_feat_ext is None):
            for p in path_test:
                os.system('cp ' + p + ' ' + path_out + '/TEST/' + lab ) 
                
            for p in path_train:
                os.system('cp ' + p + ' ' + path_out + '/TRAIN/' + lab)
                
            for p in path_val:
                os.system('cp ' + p + ' ' + path_out + '/VAL/' + lab )
        else:
            for p in path_test:
                os.system('cp ' + p + ' ' + path_out + '/TEST/' + lab ) 
                os.system('cp ' + p.split('.')[0] + '_feat.' + scalar_feat_ext + ' ' + path_out + '/TEST/' + lab ) 
                
                
            for p in path_train:
                os.system('cp ' + p + ' ' + path_out + '/TRAIN/' + lab)
                os.system('cp ' + p.split('.')[0] + '_feat.' + scalar_feat_ext + ' ' + path_out + '/TRAIN/' + lab)
                
            for p in path_val:
                os.system('cp ' + p + ' ' + path_out + '/VAL/' + lab )
                os.system('cp ' + p.split('.')[0] + '_feat.' + scalar_feat_ext + ' ' + path_out + '/VAL/' + lab )
            
            
            
            
            
            
            
            