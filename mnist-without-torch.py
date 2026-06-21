import numpy as np

epoch = 500
learning_rate = 0.1

data = np.genfromtxt('train.csv', delimiter=',', skip_header=1) 

np.random.seed(42)
np.random.shuffle(data)

train_labels = np.eye(10)[data[0:3900, 0].astype(int)]
test_labels = data[3900:, 0].astype(int)

train_data = data[0:3900, 1:] / 255.0
test_data = data[3900:, 1:] / 255.0

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class hidden_layers:
    def __init__(self, IN, out):
        self.IN = IN
        self.out = out
        self.weights = np.random.randn(IN, out) * np.sqrt(1.0 / IN)
        self.bias = np.zeros((1,out))
        self.activation = None
        self.gradient = None
        self.bgradient = None

    def set_activation(self, activation):
        self.activation = activation

    def softmax(self):
        e_x = np.exp(self.activation - np.max(self.activation, axis=-1, keepdims=True))
        self.activation = e_x / np.sum(e_x, axis=-1, keepdims=True)

layer1 = hidden_layers(train_data.shape[1], 128)
layer2 = hidden_layers(128, 10)
 
def forward_pass(x):

    z1 = sigmoid(x @ layer1.weights + layer1.bias)
    layer1.set_activation(z1)

    z2 = layer1.activation@layer2.weights + layer2.bias
    layer2.set_activation(z2) 
    layer2.softmax()
      
    return layer2.activation

def backward_pass():

    error2 = (output - train_labels)
    error1 = (( error2 @ layer2.weights.T) * layer1.activation * (1 - layer1.activation))

    layer2.bgradient = np.sum(error2, axis=0, keepdims=True) / train_data.shape[0]
    layer1.bgradient = np.sum(error1, axis=0, keepdims=True) / train_data.shape[0]

    layer2.gradient = ((layer1.activation.T) @ error2) / train_data.shape[0]
    layer1.gradient = (train_data.T @ error1) / train_data.shape[0]

def test():
    test_predictions = np.argmax(forward_pass(test_data), axis=1)
    accuracy = np.mean(test_predictions == test_labels) * 100
    print(f"Accuracy: {accuracy:.2f}%")

#training
for i in range(epoch):
    output = forward_pass(train_data)
    backward_pass()

    layer2.weights -= layer2.gradient * learning_rate
    layer1.weights -= layer1.gradient * learning_rate

    layer2.bias -= layer2.bgradient * learning_rate
    layer1.bias -= layer1.bgradient * learning_rate

test()





