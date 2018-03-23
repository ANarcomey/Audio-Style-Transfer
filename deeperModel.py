import numpy as np
import tensorflow as tf

from audio_utils import *


class deeperModel:

	def __init__(self, usingKickStart, content_tf, style_tf, N_BINS):
		self.usingKickStart = usingKickStart
		self.content_tf = content_tf
		self.style_tf = style_tf
		self.N_BINS = N_BINS

		self.N_FILTERS = [128, 256, 512]

		self.styleWeight = 1
		self.contentWeight = 0.1
		self.learning_rate = 1e-3

		self.saver = None

		self.contentInitialization = True



	def initializeWeights(self):

		N_FILTERS1 = self.N_FILTERS[0]
		N_FILTERS2 = self.N_FILTERS[1]
		N_FILTERS3 = self.N_FILTERS[2]

		filterLength = 11

		std1 = np.sqrt(2) * np.sqrt(2.0 / ((self.N_BINS + N_FILTERS1) * filterLength))
		W1_init = np.random.randn(1, filterLength, self.N_BINS, N_FILTERS1)*std1
		W1 = tf.constant(W1_init, name = 'W1', dtype = tf.float32)

		std2 = np.sqrt(2) * np.sqrt(2.0 / ((N_FILTERS1 + N_FILTERS2) * filterLength))
		W2_init = np.random.randn(1, filterLength, N_FILTERS1, N_FILTERS2)*std2
		W2 = tf.constant(W2_init, name = 'W2', dtype = tf.float32)

		std3 = np.sqrt(2) * np.sqrt(2.0 / ((N_FILTERS2 + N_FILTERS3) * filterLength))
		W3_init = np.random.randn(1, filterLength, N_FILTERS2, N_FILTERS3)*std2
		W3 = tf.constant(W3_init, name = 'W3', dtype = tf.float32)

		return W1, W2, W3

	#this function gets the style loss for model 2
	def getStyleLoss(self, activationsList):

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


	#this function gets the cost tensor for model 1
	def getCostTensor(self, Weights, OUT):
		W1,W2,W3 = Weights

		Z1_OUT = tf.nn.conv2d(OUT,W1, strides = [1,1,1,1], padding = 'SAME')
		#note: input tensor is shape [batch, in_height, in_width, in_channels]
		#kernel is shape [filter_height, filter_width, in_channels, out_channels]
		#stride according to dimensions of input

		A1_OUT = tf.nn.relu(Z1_OUT)

		Z2_OUT = tf.nn.conv2d(A1_OUT, W2, strides = [1,1,1,1], padding = 'SAME')
		A2_OUT = tf.nn.relu(Z2_OUT)

		Z3_OUT = tf.nn.conv2d(A2_OUT, W3, strides = [1,1,1,1], padding = 'SAME')
		A3_OUT = tf.nn.relu(Z3_OUT)

		Z1_content = tf.nn.conv2d(self.content_tf, W1, strides = [1,1,1,1], padding = 'SAME')
		A1_content = tf.nn.relu(Z1_content)
		Z1_style = tf.nn.conv2d(self.style_tf, W1, strides = [1,1,1,1], padding = 'SAME')
		A1_style = tf.nn.relu(Z1_style)

		Z2_content = tf.nn.conv2d(A1_content, W2, strides = [1,1,1,1], padding = 'SAME')
		A2_content = tf.nn.relu(Z2_content)
		Z2_style = tf.nn.conv2d(A1_style, W2, strides = [1,1,1,1], padding = 'SAME')
		A2_style = tf.nn.relu(Z2_style)

		Z3_content = tf.nn.conv2d(A2_content, W3, strides = [1,1,1,1], padding = 'SAME')
		A3_content = tf.nn.relu(Z3_content)
		Z3_style = tf.nn.conv2d(A2_style, W3, strides = [1,1,1,1], padding = 'SAME')
		A3_style = tf.nn.relu(Z3_style)

		styleLoss = self.getStyleLoss([(A1_OUT,A1_style), (A2_OUT, A2_style), (A3_OUT, A3_style)])

		contentLoss = 2*tf.nn.l2_loss(A3_OUT - A3_content)

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

			meta_file = 'savedModels/deeperModel/model.meta'
			checkpoint_directory = 'savedModels/deeperModel/'

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
		Weights = self.initializeWeights()
			
		cost = self.getCostTensor(Weights, OUT)


		optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(cost)


		costs = []
		shortCosts = []
		result = None

		self.saver = tf.train.Saver(max_to_keep = 4)

		with tf.Session() as session: 
			init = tf.global_variables_initializer()
			session.run(init)

			self.saver.save(session, "savedModels/deeperModel/model")


			for iteration in range(10000):
				print("new iteration, iter = ", iteration)
				_, currCost = session.run([optimizer, cost], feed_dict = {})

				print("current cost = ", currCost)
				costs.append(currCost)

				if currCost < 1e-6:
					break

				if (iteration % 500) == 0:
					print("saved model")
					self.saver.save(session, 'savedModels/deeperModel/model', write_meta_graph=False)

					shortCosts.append(currCost)
					writeListToFile(shortCosts, "shortCosts.txt")
					writeListToFile(costs, "costs.txt")



			result = session.run(OUT)

		return result