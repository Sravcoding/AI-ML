#Expected accuracy is around 97-99% for this simple CNN model after 20 epochs.
#Best value I got is 98.98%

import torch
import torchvision

epoch = 20  # number of learning cycles
learning_rate = 0.001  # learning rate
batch_size = 64  # number of samples per batch

print(f'Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')

class DigitClassifier(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.con1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) #One image input, 32 filters

        '''Kernel size: size of the filter (3x3)
        Stride: step size for moving the filter (1 pixel)
        Padding: adding pixels (0 value) around the image (1 pixel) to maintain spatial dimensions'''

        self.con2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) #32 input channels from previous layer, 64 filters
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0) #Max pooling layer to reduce spatial dimensions #2x2 pooling
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1) 

    def forward(self, x):

        x = self.relu(self.con1(x))  # Convolutional layer 1 with ReLU activation
        x = self.pool(x)  # Max pooling

        x = self.relu(self.con2(x))  # Convolutional layer 2 with ReLU activation
        x = self.pool(x)  # Max pooling

        x = x.view(-1, 64 * 7 * 7)  # Flatten the input for fully connected layers #64 channels, 7x7 image size after pooling

        x = self.relu(self.fc1(x))  # Hidden layer with ReLU activation

        x = self.fc2(x)  # Output layer

        x = self.softmax(x)  # Softmax activation for multi-class classification
        return x
    

transform_fn = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.13,), (0.30,))]) 

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform_fn, download=True) # Download MNIST dataset from torchvision (train set)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform_fn, download=True) # Test dataset
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # Create data loader for batching and shuffling training data
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # Test data loader

# Initialize model, loss function, and optimizer 
model = DigitClassifier()
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# Training loop
for e in range(epoch):
    for images, labels in train_loader:
        optimizer.zero_grad() 
        outputs = model(images) 
        loss = loss_fn(outputs, labels)  
        loss.backward() 
        optimizer.step()  

    print(f'Epoch {e+1}/{epoch}, Loss: {loss}')

# Evaluate the model on test data
model.eval()  
correct = 0
total = 0
with torch.no_grad(): 
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) 
        total += labels.size(0) 
        correct += (predicted == labels).sum().item()  

print(f'Accuracy: {100 * correct / total}%')



