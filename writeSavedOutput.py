import argparse

import tensorflow as tf
import numpy as np
import librosa

from audio_utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--output_file', help = "output file to write to. Defaults to out.wav", default = 'out.wav')
parser.add_argument('--content_file', help = "content file in order to determine sampling rate of output")




if __name__ == '__main__':



    args = parser.parse_args()
    content_file = args.content_file
    output_file = args.output_file
    meta_file = 'savedModels/shallowModel/model.meta'
    checkpoint_directory = 'savedModels/shallowModel/'

    result = getSavedOutput(meta_file, checkpoint_directory)

    #_, sampleRate = librosa.load(content_file)
    #spectro, sampleRate = read_audio_spectrum(content_file, N_FFT = 2048)
    spectro, sampleRate = read_audio_spectrum(content_file, N_FFT = 512)
    shape = spectro.shape
    print("shape = ", shape)

    writeOutput(result, sampleRate, output_file)
    #writeOutput2(result, sampleRate, output_file, shape)
    print("output written to \"" + output_file + "\"")