All saved outputs are in the folder 'Final Outputs', the folder labelled 1024 filters has the noisy results characteristic
of 1024 filters. The other two outputs are the best results from tuning the content and style weights on 4096 filters.
All hyperparameters are contained in 'out.txt' within the folder for each saved output.

There are 6 .wav files in the repository which are samples taken from the IRMAS dataset.

Note that shallowModel2 was an experimental model that did not reshape the spectrograms into 1 by N_timesteps image with the
frequency bands as channels. It took the spectrogram simply as a 2D image. Synthesizing outputs was far too slow so this
model was not used for any discussed results

Note: there is a styleTransfer.py and a jupyter notebook which both call the same code to run style tranfer models, so
use the python file if command line is more convenient or the notebook if that is more convenient. Keep in mind these models
take upwards of an hour to synthesize outputs
