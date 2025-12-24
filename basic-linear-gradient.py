#ML to predict values of w and b 

import torch

epochs = 100000 #no of learning cycles
learning_rate = 0.01 #how much we change per cycle

true_w = torch.tensor([[2.0]]) #Values to predict #true_w is 2D for matrix multiplication with x 
true_b = torch.tensor(1.0)

x = torch.rand(10,1) #one input one output

true_y = x@true_w + true_b + torch.randn(10,1)*0.1 #Equation is y=Wx + b + noise #FINAL ANSWER

guess_w = torch.randn(1,1,requires_grad=True) #initial horrible guesses at random
guess_b = torch.randn(1,requires_grad=True)

for i in range(epochs):

    guess_y = x@guess_w + guess_b

    loss = torch.mean((true_y-guess_y)**2) #Mean squared error

    loss.backward() #Calculates gradient of all tensors who we've mentioned needs grad with loss (dL/dw and dL/db) and stores in .grad

    with torch.no_grad():
        guess_w -= learning_rate*guess_w.grad #Edits our guesses based on how loss changed wrt our changes
        guess_b -= learning_rate*guess_b.grad #Uses -= as a new tensor will be created in t = t - change

    guess_w.grad.zero_() #Sets both gradients to zero for cycle repeat
    guess_b.grad.zero_() 

print(loss) #should go to zero

