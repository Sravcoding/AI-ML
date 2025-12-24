import torch
import torch.nn as nn
import torch.optim as optim

epoch = 10000 #no of learning cycles
learning_rate = 0.01 #how much we change per cycle

true_w = torch.tensor([[2.0]]) #Values to predict #true_w is 2D for matrix multiplication with x 
true_b = torch.tensor(1.0)

x = torch.rand(10,1) #one input one output

true_y = x@true_w + true_b + torch.randn(10,1)*0.1 #Equation is y=Wx + b + noise #FINAL ANSWER

class LinearRegression(nn.Module): #Basic y=mx+c layer #Inheriting from nn.Module parent

    def __init__(self):
        super().__init__()
        self.linear_layer = nn.Linear(1,1)

    def forward(self,x):
        return self.linear_layer(x)
    
model = LinearRegression() #Creates the model obj
    
optimizer = optim.Adam(model.parameters(), learning_rate) #y(t+1) = y - e*dL/dw

loss_fn = nn.MSELoss() #loss = mean(sqr(y_true - y_guess))

for i in range(epoch):
    
    guess_y = model(x) #By default model(x) is the forward function

    loss = loss_fn(true_y,guess_y) #Calculates loss

    optimizer.zero_grad() #Zeroes out the gradients

    loss.backward() #Calculates new gradients
    
    optimizer.step() #Adjusts values

print(loss)


