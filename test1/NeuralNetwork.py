
import numpy as np 
import scipy.special # sigmoid函数

# neural network class defination
class neuralNetwork:
    # initialise the neural network
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # set number of nodes in each input, hidden, output layer
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes
        
        # learning rate
        self.lr = learningRate
        
        # normal distribution / Gaussian distribution
        # 平均值为0， 标准方差为节点传入链接数目的开方，即1/sqrt（传入链接数目）
        self.weigthsIH = np.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
        self.weigthsHO = np.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))
        self.activation_function = lambda x: scipy.special.expit(x)

        pass
    
    # train the neural network
    def train(self, inputs_list, targets_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T
        
        # calculate signals into hidden layer
        hidden_inputs = np.dot(self.weigthsIH, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = np.dot(self.weigthsHO, hidden_outputs)
        # calculate the signals emerging from final outpt layer
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.weigthsHO.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.weigthsHO += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.weigthsIH += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
        pass
    
    # query the neural network
    # 接受神经网络的输入，返回网络的输出
    def query(self, inputs_list):
        # convert inputs list to 2d array
        inputs = np.array(inputs_list, ndmin = 2).T
        
        #calculate signals into hidden layer
        hidden_inputs = np.dot(self.weigthsIH, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = np.dot(self.weigthsHO, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs 
    
