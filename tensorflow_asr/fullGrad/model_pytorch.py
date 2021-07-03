import os

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from tensorflow_asr.fullGrad.misc_functions import *
from tensorflow_asr.fullGrad.smooth_fullgrad import SmoothFullGrad


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = PATH + 'dataset/'

batch_size = 1

sample_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(dataset, transform=transforms.Compose([
                       transforms.Resize((32,32)),
                       transforms.ToTensor(),
                       transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                        std = [0.229, 0.224, 0.225])
                   ])),
    batch_size= batch_size, shuffle=False)

unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])

net = Net()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(sample_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

saliency_methods = {
# FullGrad-based methods
'smooth_fullgrad': SmoothFullGrad(net)
# Other saliency methods from literature
}

def compute_saliency_and_save():
    for batch_idx, (data, _) in enumerate(sample_loader):
        data = data.to('cpu').requires_grad_()

        # Compute saliency maps for the input data
        for s in saliency_methods:
            saliency_map = saliency_methods[s].saliency(data)

            # Save saliency maps
            for i in range(data.size(0)):
                filename = save_path + str( (batch_idx+1) * (i+1))
                image = unnormalize(data[i].cpu())
                save_saliency_map(image, saliency_map[i], filename + '_' + s + '.jpg')


if __name__ == "__main__":
    # Create folder to saliency maps
    m = SmoothFullGrad(net)
    save_path = PATH + 'results/'
    create_folder(save_path)
    compute_saliency_and_save()
    print('Saliency maps saved.')


print('Finished Training')