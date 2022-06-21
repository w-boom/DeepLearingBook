import numpy as np 
# import matplotlib.pyplot
import pickle
import NeuralNetwork


def main(): 
    # number of input, hidden and output nodes
    inputNodes = 784 # 自己理解就是特征值的个数
    hiddenNodes = 100 # 经过多少次的训练，这种就类似于主成分分析，强制网络尝试总结输入的主要特点
    outputNodes = 10 # 标签数
    
    # learning rate is 0.3
    learningRate = 0.3
    
    # create instance of neural network
    n = NeuralNetwork.neuralNetwork(inputNodes, hiddenNodes, outputNodes, learningRate)
    
    with open(r'mnist_dataset/mnist.pkl', 'rb') as fid:
      data = pickle.load(fid) 
      pass
    
    trainImage = data['train_img']
    trainLabel = data['train_label']
    testImage = data['test_img']
    testLabel = data['test_label']
    
    # train the neural network
    for num in range(trainImage.size // 784):
        inputs = (trainImage[num] / 255.0 * 0.99) + 0.01
        targets = np.zeros(outputNodes) + 0.01
        targets[int(trainLabel[num])] = 0.99
        n.train(inputs, targets)
        pass
    
    # 测试
    scoreCard = [];
    for num in range(testImage.size // 784):
        correctLabel = int(testLabel[num])
        print(correctLabel, "correct label")
        inputs = (testImage[num] / 255.0 * 0.99) + 0.01
        outputs = n.query(inputs)
        label = np.argmax(outputs)
        print(label, "network's answer")
        if (label == correctLabel):
            scoreCard.append(1)
        else:
            scoreCard.append(0)
            pass
        pass
    
    # calculate the performance score, the fraction of correct answers
    scoreCardArray = np.asarray(scoreCard)
    print("performance = ", scoreCardArray.sum() / scoreCardArray.size)
    

if __name__ == '__main__':
    main()