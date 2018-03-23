import argparse

import tensorflow as tf
import numpy as np
import librosa

from shallowModel import shallowModel
from deeperModel import deeperModel
from audio_utils import *
from shallowModel2 import shallowModel2






####################### END FILE FUNCTIONS

#this function initializes the output, using a kickstart file if that is included, or
#using the content signal as initialization
def initializeOutput(kickStartFile, content_tf):
    OUT = None

    if kickStartFile == None:
        print("using content as initialization of output")
        shape = content_tf.get_shape().as_list()
        print("shape = ", shape)
        init = np.random.randn(shape[0], shape[1], shape[2], shape[3])*1e-3
        print("init = ", init)
        OUT = tf.get_variable("OUT", initializer = content_tf, dtype = tf.float32)
    else:
        print("using kickstart file as initialization of output")
        spectrogram, sampleRate = read_audio_spectrum(kickStartFile)
        N_bins, N_timesteps = spectrogram.shape
        init = np.reshape(spectrogram.T, (1,1,N_timesteps, N_bins))

        #OUT_init = np.random.randn(1,1,N_timestepsC, N_binsC)*1e-3
        OUT = tf.get_variable("OUT", initializer = tf.constant(init, dtype = tf.float32), dtype = tf.float32)

    return OUT






############################### BEGIN MODEL 2 FUNCTIONS
#Model 2 is a 3 layer convolutional neural network with 1 by 11 filters, allowing
#named constants to change the number of filters in each layer

#this function randomly initializes all filter weights for model 2
def initializeWeightsModel2(N_BINS, N_FILTERS_1, N_FILTERS_2, N_FILTERS_3):

    filterLength = 11

    std1 = np.sqrt(2) * np.sqrt(2.0 / ((N_BINS + N_FILTERS_1) * filterLength))
    W1_init = np.random.randn(1, filterLength, N_BINS, N_FILTERS_1)*std1
    W1 = tf.constant(W1_init, name = 'W1', dtype = tf.float32)

    std2 = np.sqrt(2) * np.sqrt(2.0 / ((N_FILTERS_1 + N_FILTERS_2) * filterLength))
    W2_init = np.random.randn(1, filterLength, N_FILTERS_1, N_FILTERS_2)*std2
    W2 = tf.constant(W2_init, name = 'W2', dtype = tf.float32)

    std3 = np.sqrt(2) * np.sqrt(2.0 / ((N_FILTERS_2 + N_FILTERS_3) * filterLength))
    W3_init = np.random.randn(1, filterLength, N_FILTERS_2, N_FILTERS_3)*std2
    W3 = tf.constant(W3_init, name = 'W3', dtype = tf.float32)

    return W1, W2, W3

#this function gets the style loss for model 2
def getStyleLossModel2(activationsList):

    styleLoss = 0

    for A_OUT, A_style in activationsList:
        squeezeStyle = tf.squeeze(A_style)
        squeezeOut = tf.squeeze(A_OUT)

        print("squeeze style shape = ", squeezeStyle.shape)
        s2 = tf.transpose(squeezeStyle)
        o2 = tf.transpose(squeezeOut)
        print("s2 tranpsose shape = ", s2.shape)
        Gs = tf.matmul(s2, tf.transpose(s2))
        Go = tf.matmul(o2, tf.transpose(o2))
        print("gs shape = ", Gs.shape)

        vectorStyle = tf.reshape(squeezeStyle, [-1,1])
        vectorOut = tf.reshape(squeezeOut, [-1,1])

        styleGramMatrix = tf.matmul(tf.transpose(vectorStyle), vectorStyle)
        outGramMatrix = tf.matmul(tf.transpose(vectorOut), vectorOut)

        styleGramMatrix = Gs
        outGramMatrix = Go

        styleLoss += 2*tf.nn.l2_loss(styleGramMatrix - outGramMatrix)

    return styleLoss/float(len(activationsList))

#this function gets the cost tensor for model 2
def getCostTensorModel2(OUT, content_tf, style_tf, ALPHA, BETA, W1, W2, W3):

    Z1_OUT = tf.nn.conv2d(OUT,W1, strides = [1,1,1,1], padding = 'SAME')
    #note: input tensor is shape [batch, in_height, in_width, in_channels]
    #kernel is shape [filter_height, filter_width, in_channels, out_channels]
    #stride according to dimensions of input

    A1_OUT = tf.nn.relu(Z1_OUT)

    Z2_OUT = tf.nn.conv2d(A1_OUT, W2, strides = [1,1,1,1], padding = 'SAME')
    A2_OUT = tf.nn.relu(Z2_OUT)

    Z3_OUT = tf.nn.conv2d(A2_OUT, W3, strides = [1,1,1,1], padding = 'SAME')
    A3_OUT = tf.nn.relu(Z3_OUT)

    Z1_content = tf.nn.conv2d(content_tf, W1, strides = [1,1,1,1], padding = 'SAME')
    A1_content = tf.nn.relu(Z1_content)
    Z1_style = tf.nn.conv2d(style_tf, W1, strides = [1,1,1,1], padding = 'SAME')
    A1_style = tf.nn.relu(Z1_style)

    Z2_content = tf.nn.conv2d(A1_content, W2, strides = [1,1,1,1], padding = 'SAME')
    A2_content = tf.nn.relu(Z2_content)
    Z2_style = tf.nn.conv2d(A1_style, W2, strides = [1,1,1,1], padding = 'SAME')
    A2_style = tf.nn.relu(Z2_style)

    Z3_content = tf.nn.conv2d(A2_content, W3, strides = [1,1,1,1], padding = 'SAME')
    A3_content = tf.nn.relu(Z3_content)
    Z3_style = tf.nn.conv2d(A2_style, W3, strides = [1,1,1,1], padding = 'SAME')
    A3_style = tf.nn.relu(Z3_style)

    styleLoss = getStyleLossModel2([(A1_OUT,A1_style), (A2_OUT, A2_style), (A3_OUT, A3_style)])

    contentLoss = 2*tf.nn.l2_loss(A3_OUT - A3_content)

    cost = ALPHA*styleLoss + BETA*contentLoss

    return cost


#this function implements the multi-layer model 2
def model2(kickStartFile, content_tf, style_tf, N_binsC):
    #N_FILTERS_1 = 256
    #N_FILTERS_2 = 512
    #N_FILTERS_3 = 1024

    N_FILTERS_1 = 256
    N_FILTERS_2 = 512
    N_FILTERS_3 = 1024

    N_FILTERS_1 = 64
    N_FILTERS_2 = 128
    N_FILTERS_3 = 128

    ALPHA = 1
    BETA = 0
    learning_rate = 1e-3

    OUT = initializeOutput(kickStartFile, content_tf)

    W1,W2,W3 = initializeWeightsModel2(N_binsC, N_FILTERS_1, N_FILTERS_2, N_FILTERS_3)

    cost = getCostTensorModel2(OUT, content_tf, style_tf, ALPHA, BETA, W1, W2, W3)


    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)


    costs = []
    result = None

    with tf.Session() as session: 
        init = tf.global_variables_initializer()
        session.run(init)


        for iteration in range(6000):
            print("new iteration, iter = ", iteration)
            _, currCost = session.run([optimizer, cost], feed_dict = {})

            print("current cost = ", currCost)
            costs.append(currCost)

            if currCost < 1e-6:
                break



        result = session.run(OUT)


    writeOutput(result, content_sr)

############################# END MODEL 2 FUNCTIONS




############################ MAIN FUNCTIONS
parser = argparse.ArgumentParser()

parser.add_argument('--content_file', help = ".mp3 file containing content to synthesize")
parser.add_argument('--style_file', help = ".mp3 file containing style template")
parser.add_argument('--kick_start', help = "Optional, \"True\" means use saved output from savedOutputs directory. Default is not using saved output", default = "False")


if __name__ == '__main__':




    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    args = parser.parse_args()
    contentFile = args.content_file
    styleFile = args.style_file
    kick_start_string = args.kick_start 

    usingKickStart = False
    if kick_start_string == "True":
        usingKickStart = True

    print("using kickstart = ", usingKickStart)

    inputs = {"content file":contentFile, "style file":styleFile}
    writeParamsToFile(inputs, "inputs.txt")


    contentSpectro, content_sr = read_audio_spectrum(contentFile, N_FFT = 2048)
    styleSpectro, style_sr = read_audio_spectrum(styleFile, N_FFT = 2048)

    print ("content spectrogram shape = ", contentSpectro.shape)
    print("style spectrogram shape = ", styleSpectro.shape)
    N_binsC, N_timestepsC = contentSpectro.shape
    N_binsS, N_timestepsS = styleSpectro.shape

    #In order to ensure that content and style have same shape, 
    #clips should be 10s so this shouldn't be issue
    styleSpectro = styleSpectro[:N_binsC,:N_timestepsC]


    
    content = np.reshape(contentSpectro.T, (1,1,N_timestepsC, N_binsC))
    style = np.reshape(styleSpectro.T, (1,1,N_timestepsS, N_binsS))

    content_tf = tf.constant(content, name = 'content_tf', dtype = tf.float32)
    style_tf = tf.constant(style, name = 'style_tf', dtype = tf.float32)


    model1 = shallowModel(usingKickStart, content_tf, style_tf, N_binsC)
    print('using shallow model')
    result = model1.synthesize()

    writeOutput(result, content_sr, filename = "out.wav")
    
    
    
    #model 2
    '''model2 = deeperModel(usingKickStart, content_tf, style_tf, N_binsC)
    result = model2.synthesize()

    writeOutput(result, content_sr, filename = "out.wav")'''

