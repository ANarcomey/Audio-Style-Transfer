import argparse
import os

import tensorflow as tf
import numpy as np
import librosa

from audio_utils import *


N_Bins = 1025
N_Timesteps = 130
N_Examples = 6705
N_Instruments = 11

exampleIndexToFileName = {}



class trainedShallowModel:

    def __init__(self, X, Y, analysisTools):

        self.N_FILTERS = 4096
        self.learning_rate = 0.01
        self.saver = None

        self.analysisTools = analysisTools

        assert X.shape[0] == Y.shape[0]
        self.N_Examples = X.shape[0]
        self.N_Timesteps = X.shape[2]
        self.N_Bins = X.shape[3]
        self.N_Instruments = Y.shape[1]


        testSetSize = 100
        self.X = X[:-testSetSize, :, :, :]
        self.Y = Y[:-testSetSize, :]
        self.testX = X[-testSetSize:, :, :, :]
        self.testY = Y[-testSetSize:, :]
        
        #clip recordings to 2 seconds, consistent with rest of project
        self.X = self.X[:,:,:89,:]
        self.testX = self.testX[:,:,:89,:]
        self.N_Timesteps = 89
                

        #shape of X =  (6705, 1, 130, 1025)
        #shape of Y =  (6705, 11)

        X_shape = (None, 1, self.N_Timesteps, self.N_Bins)
        Y_shape = (None, self.N_Instruments)

        self.X_tf = tf.placeholder(name = 'X_tf', shape = X_shape, dtype = tf.float32)
        self.Y_tf = tf.placeholder(name = 'Y_tf', shape = Y_shape, dtype = tf.float32)
                
        self.numTrainingExamplesUsed = 400
        self.numIterations = 1000
        
        self.minibatch_size = 4
        num_epochs = self.num_epochs = 10
        
        params = {"testSetSize":testSetSize}
        params["numTrainingExamplesUsed"] = self.numTrainingExamplesUsed
        params["learning rate"] = self.learning_rate
        params["numIterations"] = self.numIterations
        params["N_bins"] = self.N_Bins
        params["N_filters"] = self.N_FILTERS
        params["minibatch size"] = self.minibatch_size
        params["num_epochs"] = self.num_epochs
        
        writeParamsToFile(params, "IRMAS/out.txt")
        
        writeSingleParamToTxt(self.learning_rate, "setParams/learning_rate.txt")
        
        self.W = self.initializeWeights()

        fullyConnected, cost = self.getCostTensor(self.W)
        self.fc = fullyConnected
        self.cost_tf = cost


        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost)






    def initializeWeights(self):
        filterLength = 11
        shape = [1, filterLength, self.N_Bins, self.N_FILTERS]

        W = tf.get_variable("W", shape = shape, initializer = tf.contrib.layers.xavier_initializer(seed=0), dtype = tf.float32)

        return W

    def getCostTensor(self, W):

        Z = tf.nn.conv2d(self.X_tf, W, strides = [1,1,1,1], padding = 'VALID')
        #note: input tensor is shape [batch, in_height, in_width, in_channels]
        #kernel is shape [filter_height, filter_width, in_channels, out_channels]
        #stride according to dimensions of input

        A = tf.nn.relu(Z)

        flattened = tf.contrib.layers.flatten(A)
        fullyConnected =  tf.contrib.layers.fully_connected(flattened, num_outputs = self.N_Instruments, activation_fn = None, scope = 'fc')
        
        print("fc name = ", fullyConnected.name)

        softmax_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.Y_tf, logits = fullyConnected, name = 'softmax')

        cost = tf.reduce_mean(softmax_loss)


        return fullyConnected, cost

    def trainTestEval(self, session):        
        
        feedTest = {self.X_tf:self.testX, self.Y_tf:self.testY}
        X_train = self.X[:self.numTrainingExamplesUsed,:,:,:]
        Y_train = self.Y[:self.numTrainingExamplesUsed,:]
        feedTrain = {self.X_tf:X_train, self.Y_tf:Y_train}
        
        graph = tf.get_default_graph()
        #W_np = loadSavedNumpyArrays("savedModels/longerTrainedShallow/Weights.npy")
        #W_tf = graph.get_tensor_by_name("W:0")
        #zeros = (np.zeros([1, 11, self.N_Bins, self.N_FILTERS]))
        
        #feedTrain[W_tf] = zeros
        
        #print("W = ", W)
        #print("self.W = ", self.W)
        fc = graph.get_tensor_by_name("fc/BiasAdd:0")
        #print("fc = ", session.run(fc, feed_dict = feedTest))


        predict = tf.argmax(fc, 1)
        correct_prediction = tf.equal(predict, tf.argmax(self.Y_tf, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        #print("test correct pred = ", correct_prediction.eval(feed_dict = feedTest))
        print("test accuracy = ", session.run(accuracy, feed_dict = feedTest))
        
        print("train accuracy = ", session.run(accuracy, feed_dict = feedTrain))
        #print("train correct pred = ", correct_prediction.eval(feed_dict = feedTrain))
        
  

    def trainWithMinibatch(self):
        
        minibatch_size = self.minibatch_size
        num_epochs = self.num_epochs
        


        costs = []
        shortCosts = []
        result = None

        self.saver = tf.train.Saver(max_to_keep = 4)
        X_train = self.X[:self.numTrainingExamplesUsed,:,:,:]
        Y_train = self.Y[:self.numTrainingExamplesUsed,:]
        m = self.numTrainingExamplesUsed
        
        with tf.Session() as session: 
            init = tf.global_variables_initializer()
            session.run(init)

            self.saver.save(session, "savedModels/trainedShallowModel/model")


            for epoch in range(num_epochs):
                print("new epoch, epoch = ", epoch)

                minibatch_cost = 0.
                num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
                minibatches = random_mini_batches(X_train, Y_train, minibatch_size)
                
                minibatchCounter = 0
        
                for minibatch in minibatches:
                    print("")
                    print("new minibatch, # ", minibatchCounter)
                    minibatchCounter += 1

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    # IMPORTANT: The line that runs the graph on a minibatch.
                    # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                    ### START CODE HERE ### (1 line)
                    _ , temp_cost = session.run([self.optimizer, self.cost_tf], feed_dict={self.X_tf: minibatch_X, self.Y_tf: minibatch_Y})
                    ### END CODE HERE ###

                    minibatch_cost += temp_cost / num_minibatches
                    self.trainTestEval(session)
                    
                print("current cost = ", minibatch_cost)
                costs.append(minibatch_cost)
                
                


                if (epoch % 1) == 0:
                    print("saved model")
                    self.saver.save(session, 'savedModels/trainedShallowModel/model', write_meta_graph=False)

                    shortCosts.append(minibatch_cost)
                    writeListToFile(shortCosts, "savedModels/trainedShallowModel/shortCosts.txt")
                    writeListToFile(costs, "savedModels/trainedShallowModel/costs.txt")
                    
                    print("saved weights")
                    np.save("IRMAS/Weights", session.run(self.W))
                    
                    self.trainTestEval(session)
                    
            result = session.run(self.W)
                    
        return result
                    
            
    
    def train(self):


        costs = []
        shortCosts = []
        result = None

        self.saver = tf.train.Saver(max_to_keep = 4)
        feed = {self.X_tf:self.X[:self.numTrainingExamplesUsed,:,:,:], self.Y_tf:self.Y[:self.numTrainingExamplesUsed,:]}

        with tf.Session() as session: 
            init = tf.global_variables_initializer()
            session.run(init)

            self.saver.save(session, "savedModels/trainedShallowModel/model")


            for iteration in range(self.numIterations):
                print("")
                print("new iteration, iter = ", iteration)
                _, currCost = session.run([self.optimizer, self.cost_tf], feed_dict = feed)
                

                print("current cost = ", currCost)
                costs.append(currCost)
                self.trainTestEval(session)


                if (iteration % 5) == 0:
                    print("saved model")
                    self.saver.save(session, 'savedModels/trainedShallowModel/model', write_meta_graph=False)

                    shortCosts.append(currCost)
                    writeListToFile(shortCosts, "savedModels/trainedShallowModel/shortCosts.txt")
                    writeListToFile(costs, "savedModels/trainedShallowModel/costs.txt")
                    
                    #self.learning_rate = loadParamFromTxt("setParams/learning_rate.txt")
                    #optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost)
                    
                    print("saved weights")
                    np.save("IRMAS/Weights", session.run(self.W))
                    



            print("saved model")
            self.saver.save(session, 'savedModels/trainedShallowModel/model', write_meta_graph=False)

            shortCosts.append(currCost)
            writeListToFile(shortCosts, "savedModels/trainedShallowModel/shortCosts.txt")
            writeListToFile(costs, "savedModels/trainedShallowModel/costs.txt")



            result = session.run(self.W)

            self.trainTestEval(session)




        return result


def initializeShuffledXAndY(read_raw_data, use_preshuffled):
    if read_raw_data:
        print("reading in raw data")
        X,Y = readXAndY(instrumentAbbrevs, abbrevToIndex)
        print("shape of X = ", X.shape)
        print("shape of Y = ", Y.shape)

        np.save("IRMAS/savedY", Y)
        np.save("IRMAS/savedX", X)


        shuffledX, shuffledY = shuffle(X,Y)
        np.save("IRMAS/savedXShuffled", shuffledX)
        np.save("IRMAS/savedYShuffled", shuffledY)

        print("saved X and Y in 'IRMAS/savedX' and 'IRMAS/savedY'")
        print("saved shuffled X and Y in 'IRMAS/savedXShuffled' and 'IRMAS/savedYShuffled'")

    else:
        print("using saved preprocessed data")

    #shape of X =  (6705, 1, 130, 1025)
    #shape of Y =  (6705, 11)

    x_file = "IRMAS/savedX.npy"
    y_file = "IRMAS/savedY.npy"

    x_shuffled_file = "IRMAS/savedXShuffled.npy"
    y_shuffled_file = "IRMAS/savedYShuffled.npy"

    X,Y = None, None

    if use_preshuffled:
        print("using preshuffled data")
        X,Y = loadSavedNumpyArrays(filenames = (x_shuffled_file, y_shuffled_file))
    else:
        print("Shuffling data")
        X,Y = loadSavedNumpyArrays(filenames = (x_file, y_file))
        X,Y = shuffle(X,Y)
        
        np.save("IRMAS/savedXShuffled", X)
        np.save("IRMAS/savedYShuffled", Y)

    #note: shuffle data and save into shuffled file

    return X,Y


def getConversions():
    instrumentNames = loadList("IRMAS/instrumentNames.txt")
    instrumentAbbrevs = loadList("IRMAS/instrumentAbbrevs.txt")

    print("instrument names read from 'IRMAS/instrumentNames.txt'")
    print("instrument abbreviations read from 'IRMAS/instrumentAbbrevs.txt'")


    assert len(instrumentAbbrevs) == len(instrumentNames)

    abbrevToFullName = {}
    fullNameToAbbrev = {}

    indexToAbbrev = {}
    abbrevToIndex = {}
    for i in range(len(instrumentNames)):
        name = instrumentNames[i]
        abbrev = instrumentAbbrevs[i]

        abbrevToFullName[abbrev] = name
        fullNameToAbbrev[name] = abbrev

        indexToAbbrev[i] = abbrev
        abbrevToIndex[abbrev] = i

    conversions = (abbrevToFullName, fullNameToAbbrev, indexToAbbrev, abbrevToIndex)

    return conversions, instrumentNames, instrumentAbbrevs


def readXAndY(instrumentAbbrevs, abbrevToIndex):
    
    X = np.zeros((N_Examples, 1, N_Timesteps, N_Bins))
    Y = np.zeros((N_Examples, N_Instruments))

    example_index = 0
    for abbrev in instrumentAbbrevs:
        print("abbrev = ", abbrev)

        instrumentIndex = abbrevToIndex[abbrev]
        y_vec = np.zeros((N_Instruments))
        y_vec[instrumentIndex] = 1


        for filename in os.listdir("IRMAS/TrainingData/" + abbrev):
            if filename.endswith(".wav"):

                Y[example_index, :] = y_vec

                completePath = "IRMAS/TrainingData/" + abbrev + "/" + filename
                spectrogram, sampleRate = read_audio_spectrum(completePath)

                curr_N_bins, curr_N_timesteps = spectrogram.shape

                assert curr_N_bins == N_Bins and curr_N_timesteps == N_Timesteps


                reshaped = np.reshape(spectrogram.T, (1,N_Timesteps, N_Bins))
                X[example_index, :, :, :] = reshaped

                example_index += 1
                print("example_index = ", example_index)
                exampleIndexToFileName[example_index] = completePath

            else:
                print("wrong file type, not wav")

    return X,Y


parser = argparse.ArgumentParser()
parser.add_argument('--read_raw_data', help = "Optional, enter 'True' to read the raw data from scratch and generate X and Y arrays")
parser.add_argument('--preshuffled_data', help = "For debugging purposes, uses preshuffled data to save time. Enter 'True' to use")


if __name__ == '__main__':




    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    args = parser.parse_args()
    read_raw_data = args.read_raw_data
    use_preshuffled = args.preshuffled_data
    
    conversions, instrumentNames, instrumentAbbrevs = getConversions()

    abbrevToFullName, fullNameToAbbrev, indexToAbbrev, abbrevToIndex = conversions

    analysisTools = {}
    analysisTools["abbrevToFullName"] = abbrevToFullName
    analysisTools["fullNameToAbbrev"] = fullNameToAbbrev
    analysisTools["indexToAbbrev"] = indexToAbbrev
    analysisTools["abbrevToIndex"] = abbrevToIndex
    analysisTools["exampleIndexToFileName"] = exampleIndexToFileName


    X, Y = initializeShuffledXAndY(read_raw_data, use_preshuffled)
    




    model = trainedShallowModel(X,Y, analysisTools)
    #Weights = model.trainWithMinibatch()
    Weights = model.train()

    np.save("IRMAS/Weights", Weights)




