from __future__ import division
import os
import time
import math
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils

import tensorflow.contrib as tc 

from layers_slim import *



def FCN_Seg(self, is_training=True):

    #Set training hyper-parameters
    self.is_training = is_training
    self.normalizer = tc.layers.batch_norm
    self.bn_params = {'is_training': self.is_training}

      
    print("input", self.tgt_image)

    with tf.variable_scope('First_conv'):
        conv1 = tc.layers.conv2d(self.tgt_image, 32, 3, 1, normalizer_fn=self.normalizer, normalizer_params=self.bn_params)

        print("Conv1 shape")
        print(conv1.get_shape())

    x = inverted_bottleneck(conv1, 1, 16, 0,self.normalizer, self.bn_params, 1)
    #print("Conv 1")
    #print(x.get_shape())

    #180x180x24
    x = inverted_bottleneck(x, 6, 24, 1,self.normalizer, self.bn_params, 2)
    x = inverted_bottleneck(x, 6, 24, 0,self.normalizer, self.bn_params, 3)
    
    print("Block One dim ")
    print(x)

    DB2_skip_connection = x    
    #90x90x32
    x = inverted_bottleneck(x, 6, 32, 1,self.normalizer, self.bn_params, 4)
    x = inverted_bottleneck(x, 6, 32, 0,self.normalizer, self.bn_params, 5)
    
    print("Block Two dim ")
    print(x)

    DB3_skip_connection = x
    #45x45x96
    x = inverted_bottleneck(x, 6, 64, 1,self.normalizer, self.bn_params, 6)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 7)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 8)
    x = inverted_bottleneck(x, 6, 64, 0,self.normalizer, self.bn_params, 9)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 10)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 11)
    x = inverted_bottleneck(x, 6, 96, 0,self.normalizer, self.bn_params, 12)
    
    print("Block Three dim ")
    print(x)

    DB4_skip_connection = x
    #23x23x160
    x = inverted_bottleneck(x, 6, 160, 1,self.normalizer, self.bn_params, 13)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 14)
    x = inverted_bottleneck(x, 6, 160, 0,self.normalizer, self.bn_params, 15)
    
    print("Block Four dim ")
    print(x)

    #23x23x320
    x = inverted_bottleneck(x, 6, 320, 0,self.normalizer, self.bn_params, 16)
    
    print("Block Four dim ")
    print(x)
    

    # Configuration 1 - single upsampling layer
    if self.configuration == 1:

        #input is features named 'x'

        # TODO(1.1) - incorporate a upsample function which takes the features of x 
        # and produces 120 output feature maps, which are 16x bigger in resolution than 
        # x. Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up5

        current_up5 = TransitionUp_elu(x, 120, 16,  'Decoder1')
        current_up5 = crop(current_up5, self.tgt_image)

        End_maps_decoder1 = slim.conv2d(current_up5, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    # Configuration 2 - single upsampling layer plus skip connection
    if self.configuration == 2:

        #input is features named 'x'

        # TODO (2.1) - implement the refinement block which upsample the data 2x like in configuration 1 
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps 
        upsample2x = TransitionUp_elu(x, 256, 2,  'Decoder21')
        
        # crop the bigger
        cropped_upsample2x = crop(upsample2x, DB4_skip_connection)
        cropped_upsample2x = crop(DB4_skip_connection, cropped_upsample2x)

        fused_upsample2x = Concat_layers(cropped_upsample2x, DB4_skip_connection, 'fused_Decoder21')

        current_up = slim.conv2d(fused_upsample2x, 256, [3, 3], scope='conv_fused_Decoder21')
        
        # TODO (2.2) - incorporate a upsample function which takes the features from TODO (2.1) 
        # and produces 120 output feature maps, which are 8x bigger in resolution than 
        upsample8x = TransitionUp_elu(current_up, 120, 8,  'Decoder22')

        # TODO (2.1). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up3

        current_up3 = crop(upsample8x, self.tgt_image)

        End_maps_decoder1 = slim.conv2d(current_up3, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    # Configuration 3 - Two upsampling layer plus skip connection
    if self.configuration == 3:

        #input is features named 'x'

        # TODO (3.1) - implement the refinement block which upsample the data 2x like in configuration 1 
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps
        upsample2x = TransitionUp_elu(x, 256, 2,  'Decoder31')

        # crop the bigger
        cropped_upsample2x = crop(upsample2x, DB4_skip_connection)
        cropped_upsample2x = crop(DB4_skip_connection, cropped_upsample2x)

        fused_upsample2x = Concat_layers(cropped_upsample2x, DB4_skip_connection, 'fused_Decoder31')

        current_up1 = slim.conv2d(fused_upsample2x, 256, [3, 3], scope='conv_fused_Decoder31')
       
        # TODO (3.2) - Repeat TODO(3.1) now producing 160 output feature maps and fusing the upsampled features 
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.
        upsample2x2 = TransitionUp_elu(current_up1, 160, 2,  'Decoder32')

        # crop the bigger
        cropped_upsample2x = crop(upsample2x2, DB3_skip_connection)
        cropped_upsample2x = crop(DB3_skip_connection, cropped_upsample2x)

        fused_upsample2x2 = Concat_layers(cropped_upsample2x, DB3_skip_connection, 'fused_Decoder32')

        current_up2 = slim.conv2d(fused_upsample2x2, 160, [3, 3], scope='conv_fused_Decoder32')

        # TODO (3.3) - incorporate a upsample function which takes the features from TODO (3.2)  
        # and produces 120 output feature maps which are 4x bigger in resolution than 
        upsample4x = TransitionUp_elu(current_up2, 120, 4,  'Decoder33')

        # TODO (3.2). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4  

        current_up4 = crop(upsample4x, self.tgt_image)

        End_maps_decoder1 = slim.conv2d(current_up4, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)


    #Full configuration 
    if self.configuration == 4:

        ######################################################################################
        ######################################### DECODER Full #############################################

       
        
        # TODO (4.1) - implement the refinement block which upsample the data 2x like in configuration 1 
        # but that also fuse the upsampled features with the corresponding skip connection (DB4_skip_connection)
        # through concatenation. After that use a convolution with kernel 3x3 to produce 256 output feature maps 
        upsample2x = TransitionUp_elu(x, 256, 2,  'Decoder41')

        # crop the bigger
        cropped_upsample2x = crop(upsample2x, DB4_skip_connection)
        cropped_upsample2x = crop(DB4_skip_connection, cropped_upsample2x)


        fused_upsample2x = Concat_layers(cropped_upsample2x, DB4_skip_connection, 'fused_Decoder41')

        current_up1 = slim.conv2d(fused_upsample2x, 256, [3, 3], scope='conv_fused_Decoder41')
        # TODO (4.2) - Repeat TODO(4.1) now producing 160 output feature maps and fusing the upsampled features 
        # with the corresponding skip connection (DB3_skip_connection) through concatenation.
        upsample2x2 = TransitionUp_elu(current_up1, 160, 2,  'Decoder42')

        # crop the bigger
        cropped_upsample2x2 = crop(upsample2x2, DB3_skip_connection)
        cropped_upsample2x2 = crop(DB3_skip_connection, cropped_upsample2x2)

        fused_upsample2x2 = Concat_layers(cropped_upsample2x2, DB3_skip_connection, 'fused_Decoder42')

        current_up2 = slim.conv2d(fused_upsample2x2, 160, [3, 3], scope='conv_fused_Decoder42')
        # TODO (4.3) - Repeat TODO(4.2) now producing 96 output feature maps and fusing the upsampled features 
        # with the corresponding skip connection (DB2_skip_connection) through concatenation.
        upsample2x3 = TransitionUp_elu(current_up2, 96, 2,  'Decoder43')

        # crop the bigger
        cropped_upsample2x3 = crop(upsample2x3, DB2_skip_connection)
        cropped_upsample2x3 = crop(DB2_skip_connection, cropped_upsample2x3)

        fused_upsample2x3= Concat_layers(cropped_upsample2x3, DB2_skip_connection, 'fused_Decoder43')

        current_up3 = slim.conv2d(fused_upsample2x3, 96, [3, 3], scope='conv_fused_Decoder43')
        # TODO (4.4) - incorporate a upsample function which takes the features from TODO(4.3) 
        # and produce 120 output feature maps which are 2x bigger in resolution than
        upsample2x4 = TransitionUp_elu(current_up3, 120, 2,  'Decoder44')

        # TODO(4.3). Remember if dim(upsampled_features) > dim(imput image) you must crop
        # upsampled_features to the same resolution as imput image
        # output feature name should match the next convolution layer, for instance
        # current_up4
        current_up4 = crop(upsample2x4, self.tgt_image)
        
        End_maps_decoder1 = slim.conv2d(current_up4, self.N_classes, [1, 1], scope='Final_decoder') #(batchsize, width, height, N_classes)
        
        Reshaped_map = tf.reshape(End_maps_decoder1, (-1, self.N_classes))

        print("End map size Decoder: ")
        print(Reshaped_map)

    
    return Reshaped_map

