#Expected accuracy is around 95-98% for this simple model after 20 epochs.
#Best value I got is 97.4%

import torch
import torchvision

epoch = 20  # number of learning cycles
learning_rate = 0.001  # learning rate
batch_size = 64  # number of samples per batch

print(f'Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')

# Define a simple feedforward neural network for handwritten digit classification
class DigitClassifier(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(784, 128) #Input = 28*28=784 pixels, Hidden layer 1 = 128 neurons
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1) #dim=1 to apply softmax across rows (classes)

    def forward(self, x):
        x = x.view(-1, 784)  # Flatten the input #-1 means infer batch size automatically

        '''.view() is a tensor class method. it reshapes the data
        from (batch_size, 1, 28, 28) to (batch_size, 784)
        This is necessary because the fully connected layer expects a 2D input (batch_size, features)
        the images is a 4d tensor (4th dimension is individual images, 3rd is color channels, 2nd and 1st are height and width)
        here since MNIST images are grayscale, color channels = 1, hence it couldve been a 3d tensor too.
        but pytorch dataloaders give 4d tensors for images by default to deal with color images too.'''

        x = self.relu(self.fc1(x))  # Hidden layer 1 with ReLU activation
        x = self.relu(self.fc2(x))  # Hidden layer 2 with ReLU activation
        x = self.fc3(x)  # Output layer
        x = self.softmax(x)  # Softmax activation for multi-class classification
        return x
    

transform_fn = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), # Convert images to tensor in the range [0, 1]
                                               torchvision.transforms.Normalize((0.13,), (0.30,))]) # Normalize with mean and std deviation (typical for MNIST)

'''Normalization helps in faster convergence by keeping input values in a similar range which reduces 
 the zig-zagging of gradients during training and avoids unit variance issues.
 It centers the data around 0 (ie, mean 0) and scales it to have a standard deviation of 1.
 Here the first ToTensor converts pixel values from [0, 255] to [0, 1]. (ie Normalization)
 The second is Standarization which makes mean 0 and std dev 1.'''

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_fn, download=True) # Download MNIST dataset from torchvision (train set)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_fn, download=True) # Test dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # Create data loader for batching and shuffling training data
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # Test data loader

# Initialize model, loss function, and optimizer #housekeeping
model = DigitClassifier()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

for e in range(epoch):
    for images, labels in train_loader:
        #Images is the question, labels are the true answers
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

    print(f'Epoch {e+1}/{epoch}, Loss: {loss}')

# Evaluate the model on test data
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
with torch.no_grad(): # No need to compute gradients during evaluation
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) # The _ gets the max value, predicted gets the index of max value. Discard _.
        total += labels.size(0) # Total number of labels #which here is batch size
        correct += (predicted == labels).sum().item()  #predicted == labels gives a tensor of T/F, sum() counts Trues, .item() converts to int

print(f'Accuracy: {100 * correct / total}%')

