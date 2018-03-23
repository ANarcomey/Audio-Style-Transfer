import numpy as np
import tensorflow as tf

from audio_utils import *


class shallowModel:

    def __init__(self, usingKickStart, content_tf, style_tf, N_BINS):
        self.usingKickStart = usingKickStart
        self.content_tf = content_tf
        self.style_tf = style_tf
        self.N_BINS = N_BINS

        self.N_FILTERS = 4096
        #used to be 4096

        self.styleWeight = 1
        self.contentWeight = 0.01
        self.learning_rate = 1e-3

        self.saver = None

        self.contentInitialization = False

        params = {"N_filters":self.N_FILTERS, "style weight":self.styleWeight}
        params["content weight"] = self.contentWeight
        params["learning rate"] = self.learning_rate
        params["initialize with content"] = self.contentInitialization
        params["N_bins"] = self.N_BINS
        params["spectro shape"] = self.content_tf.shape.as_list()
        writeParamsToFile(params, "out.txt")
        
        writeSingleParamToTxt(self.learning_rate, "setParams/learning_rate.txt")



    def initializeWeights(self):

        filterLength = 11

        std = np.sqrt(2) * np.sqrt(2.0 / ((self.N_BINS + self.N_FILTERS) * filterLength))
        W_init = np.random.randn(1, filterLength, self.N_BINS, self.N_FILTERS)*std
        
        W_trained = loadSavedNumpyArrays(["IRMAS/TrainedWeights.npy"])[0]
        print("W_init shape = ", W_init.shape)
        print("W_trained shape = ", W_trained.shape)
        #W = tf.constant(W_init, name = 'W', dtype = tf.float32)
        W = tf.constant(W_trained, name = 'W', dtype = tf.float32)

        return W


    #this function gets the cost tensor for model 1
    def getCostTensor(self, W, OUT):

        content_tf = self.content_tf
        style_tf = self.style_tf

        Z_OUT = tf.nn.conv2d(OUT,W, strides = [1,1,1,1], padding = 'VALID')
        #note: input tensor is shape [batch, in_height, in_width, in_channels]
        #kernel is shape [filter_height, filter_width, in_channels, out_channels]
        #stride according to dimensions of input

        A_OUT = tf.nn.relu(Z_OUT)

        Z_content = tf.nn.conv2d(content_tf, W, strides = [1,1,1,1], padding = 'VALID')
        A_content = tf.nn.relu(Z_content)
        Z_style = tf.nn.conv2d(style_tf, W, strides = [1,1,1,1], padding = 'VALID')
        A_style = tf.nn.relu(Z_style)

        print("a style shape = ", A_style.shape)

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
        print("old gs shape = ", styleGramMatrix.shape)


        styleGramMatrix = Gs
        outGramMatrix = Go

        styleLoss = 2*tf.nn.l2_loss(styleGramMatrix - outGramMatrix)
        contentLoss = 2*tf.nn.l2_loss(A_OUT - A_content)

        cost = self.styleWeight*styleLoss + self.contentWeight*contentLoss

        return cost

    def initializeOutput(self):
        OUT = None

        if self.contentInitialization:
            print("using content as initialization of output")
            OUT = tf.get_variable("OUT", initializer = self.content_tf, dtype = tf.float32)
        else:
            print("using random initialization of output")
            shape = self.content_tf.get_shape().as_list()
            print("shape of content_tf = ", shape)
            init = np.random.randn(shape[0], shape[1], shape[2], shape[3])*1e-3
            OUT = tf.get_variable("OUT", initializer = tf.constant(init, dtype = tf.float32), dtype = tf.float32)

        return OUT

    #this function implements model 1 and writes the output to out.wav

    def getWeightsAndOutput(self):

        evalW, evalOut = None, None
        if self.usingKickStart:
            print("using saved model for initialization of output")

            meta_file = 'savedModels/shallowModel/model.meta'
            checkpoint_directory = 'savedModels/shallowModel/'

            with tf.Session() as session:
                print("in sess")
                saver = tf.train.import_meta_graph(meta_file)
                print("imported meta graph")
                saver.restore(session, tf.train.latest_checkpoint(checkpoint_directory))
                print("restored")

                graph = tf.get_default_graph()
                print("created graph")

                savedOUT = graph.get_tensor_by_name("OUT:0")
                savedW = graph.get_tensor_by_name("W:0")
                print("got output tensors")
                evalW = session.run(savedW)
                evalOut = session.run(savedOUT)
                #result = session.run(output)
                #print("got result")
                OUT = tf.get_variable("OUT", initializer = tf.constant(evalOut, dtype = tf.float32))
                #W = tf.get_variable("W", initializer = tf.constant(evalW, dtype = tf.float32))
                return OUT


            #savedOutput = getSavedOutput(meta_file, checkpoint_directory)
            #print("saved output = ", savedOutput.shape)
            #print("saved output type = ", type(savedOutput))
            #OUT = tf.get_variable("OUT", initializer = tf.constant(savedOutput, dtype = tf.float32), dtype = tf.float32)
            #W = 
        else:
            OUT = self.initializeOutput()
            #W = self.initializeWeights()
            return OUT

            ###note: create session, extract W and OUT, return them in funct
            #### in same funct, return regular initialize output and weights, rename to rand?


    def synthesize(self):

        OUT = self.getWeightsAndOutput()
        W = self.initializeWeights()

        cost = self.getCostTensor(W, OUT)


        optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost)


        costs = []
        shortCosts = []
        result = None

        self.saver = tf.train.Saver(max_to_keep = 4)

        with tf.Session() as session: 
            init = tf.global_variables_initializer()
            session.run(init)

            self.saver.save(session, "savedModels/shallowModel/model")


            for iteration in range(60001):
                print("new iteration, iter = ", iteration)
                _, currCost = session.run([optimizer, cost], feed_dict = {})

                print("current cost = ", currCost)
                costs.append(currCost)

                if currCost < 1e-6:
                    break

                if (iteration % 500) == 0:
                    print("saved model")
                    self.saver.save(session, 'savedModels/shallowModel/model', write_meta_graph=False)

                    shortCosts.append(currCost)
                    writeListToFile(shortCosts, "savedModels/shallowModel/shortCosts.txt")
                    writeListToFile(costs, "savedModels/shallowModel/costs.txt")
                    
                    self.learning_rate = loadParamFromTxt("setParams/learning_rate.txt")
                    
                    #optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost)
                    
                    print("learning rate = ", self.learning_rate)
                    
                    



            result = session.run(OUT)

        return result