import argparse

import tensorflow as tf
import numpy as np
import librosa

from shallowModel import shallowModel
from deeperModel import deeperModel
from audio_utils import *
from shallowModel2 import shallowModel2




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

