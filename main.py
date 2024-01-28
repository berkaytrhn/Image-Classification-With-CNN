import math
from time import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder


# create basic convolutional neural network
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ).to(setDevice())

        # nn.Dropout(0.25),
        # nn.Dropout(0.5),
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Linear(256, 15)
        ).to(setDevice())

    def forward(self, value):
        value = self.model(value)
        value = self.classifier(value)
        return value


# convolutional neural network with dropouts without residual connections
class DropMyNet(nn.Module):
    def __init__(self):
        super(DropMyNet, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        ).to(setDevice())

        # nn.Dropout(0.25),
        # nn.Dropout(0.5),
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.45),
            nn.Linear(4096, 256),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(256, 15)
        ).to(setDevice())

    def forward(self, value):
        value = self.model(value)
        value = self.classifier(value)
        return value


# convolutional neural network with residual connections
class ResidualMyNet(nn.Module):
    def __init__(self):
        super(ResidualMyNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()

        self.fc1 = nn.Linear(49152, 512)

        self.relu6 = nn.ReLU()

        self.fc2 = nn.Linear(512, 15)

    def forward(self, value):
        # conv1
        res1 = value
        out = self.conv1(value)
        out = self.relu1(out)

        # conv2
        res2 = out
        out = out + res1
        out = self.conv2(out)
        out = self.relu2(out)

        # conv3
        res3 = out
        out = out + res2
        out = self.conv3(out)
        out = self.relu3(out)

        # conv4
        res4 = out
        out = out + res3
        out = self.conv4(out)
        out = self.relu4(out)

        # conv5
        out = out + res4
        out = self.conv5(out)
        out = self.relu5(out)

        # resize
        out = out.view(out.size(0), -1)

        # fc1
        out = self.fc1(out)
        out = self.relu6(out)

        # fc2
        out = self.fc2(out)
        return out


# convolutional neural network with dropouts and residual connections
class DropResidualMyNet(nn.Module):
    def __init__(self):
        super(DropResidualMyNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()

        self.fc1 = nn.Linear(49152, 512)
        self.drop1 = nn.Dropout(0.35)

        self.relu6 = nn.ReLU()

        self.drop2 = nn.Dropout(0.35)
        self.fc2 = nn.Linear(512, 15)

    def forward(self, value):
        # conv1
        res1 = value
        out = self.conv1(value)
        out = self.relu1(out)

        # conv2
        res2 = out
        out = out + res1
        out = self.conv2(out)
        out = self.relu2(out)

        # conv3
        res3 = out
        out = out + res2
        out = self.conv3(out)
        out = self.relu3(out)

        # conv4
        res4 = out
        out = out + res3
        out = self.conv4(out)
        out = self.relu4(out)

        # conv5
        out = out + res4
        out = self.conv5(out)
        out = self.relu5(out)

        # resize
        out = out.view(out.size(0), -1)

        # fc1
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.relu6(out)

        # fc2
        out = self.drop2(out)
        out = self.fc2(out)
        return out


def defineTransforms():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomRotation(degrees=40),

            transforms.Resize(256),
            transforms.CenterCrop(128),

            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]
    )
    return transform


def loadData(testPercent, batchSize, dataset):
    # train test set separation
    trainNumber = math.floor((100 - testPercent) * len(dataset) / 100)
    testNumber = math.ceil(testPercent * len(dataset) / 100)
    trainSet, testSet = random_split(dataset, (trainNumber, testNumber), generator=torch.Generator().manual_seed(42))

    # initializing data loaders
    trainData = DataLoader(
        dataset=trainSet,
        batch_size=batchSize,
        shuffle=False,
        num_workers=4
    )

    testData = DataLoader(
        dataset=testSet,
        batch_size=batchSize,
        shuffle=False,
        num_workers=4
    )
    return trainData, testData


def setDevice():
    # set device
    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Using", device)
    return device


def initModel(newNetwork, device, path, learningRate):
    # initializing model
    model = None
    if newNetwork:
        model = DropResidualMyNet().to(device)
    else:
        model = torch.load(path, map_location=device)

    # loss and optimization function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)

    return model, criterion, optimizer


def Train(epoch, model, print_every, trainData, criterion, device, optimizer):
    # method for training model
    totalLoss = 0
    start = time()

    accuracy = []

    for i, batch in enumerate(trainData, 1):

        minput = batch[0].to(device)
        target = batch[1].to(device)

        # output of model
        moutput = model(minput)

        # computing the cross entropy loss
        loss = criterion(moutput, target)
        totalLoss += loss.item()

        optimizer.zero_grad()

        # Back propogation
        loss.backward()

        # updating model parameters
        optimizer.step()

        argmax = moutput.argmax(dim=1)
        # calculating accuracy by comparing to target
        accuracy.append((target == argmax).sum().item() / target.shape[0])

        if i % print_every == 0:
            print('Epoch: {}, Train Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(epoch, loss.item(),
                                                                                             sum(accuracy) / len(
                                                                                                 accuracy),
                                                                                             time() - start))
    # Returning Average Training Loss and Accuracy
    return totalLoss / len(trainData), sum(accuracy) / len(accuracy)


def Test(epoch, model, epochBased, testData, criterion, device):
    totalLoss = 0
    start = time()

    accuracy = []

    with torch.no_grad():  # disable calculations of gradients for all pytorch operations inside the block
        for i, batch in enumerate(testData):
            minput = batch[0].to(device)
            target = batch[1].to(device)

            # output by our model
            moutput = model(minput)

            # computing the cross entropy loss
            loss = criterion(moutput, target)
            totalLoss += loss.item()

            argmax = moutput.argmax(dim=1)
            # Find the accuracy of the batch by comparing it with actual targets
            accuracy.append((target == argmax).sum().item() / target.shape[0])

    if epochBased:
        print('Epoch: [{}], Test Loss: {:.4f}, Accuracy: {:.2f}, Time: {:.2f} sec'.format(epoch,
                                                                                          totalLoss / len(testData),
                                                                                          sum(accuracy) / len(accuracy),
                                                                                          time() - start))
    # Returning Average Testing Loss and Accuracy
    return totalLoss / len(testData), sum(accuracy) / len(accuracy)


def epochLoop(epochNumber, model, trainData, testData, criterion, device, optimizer, batchSize, learningRate):
    # main loop
    trainLosses = []
    testLosses = []
    trainAccuracies = []
    testAccuracies = []

    for epoch in range(1, epochNumber + 1):
        trainLoss, trainAccuracy = Train(epoch, model, 10, trainData, criterion, device, optimizer)
        testLoss, testAccuracy = Test(epoch, model, True, testData, criterion, device)

        trainLosses.append(trainLoss)
        testLosses.append(testLoss)
        trainAccuracies.append(trainAccuracy)
        testAccuracies.append(testAccuracy)

        print('\n')

        if epoch % 30 == 0:
            # SAVING MODEL TO USE IT LATER
            torch.save(model, f"./model_epoch-30_lr-{learningRate}_batch-{batchSize}_dropout-0.45.pth")
    return trainLosses, testLosses, trainAccuracies, testAccuracies


def testingModel(epoch, model, testData, criterion, device):
    loss, accuracy = Test(epoch, model, False, testData, criterion, device)
    return loss, accuracy


def plotting(train, test, text, batch, lr, status):
    plt.plot(range(1, len(train) + 1), train, 'r', label="Train {}".format(text))
    plt.plot(range(1, len(test) + 1), test, 'b', label="Test {}".format(text))

    plt.title(f"{batch} Batch and {lr} Learning Rate {status}")
    plt.xlabel('Epoch')
    plt.ylabel(text)
    plt.legend()
    plt.show()


def main():
    # START

    # hyper parameters
    batchSize = 64
    learningRate = 0.0005
    epochNumber = 30

    # defining transforms
    transform = defineTransforms()

    # dataset from directory
    dataset = ImageFolder("dataset", transform=transform)

    # load data from dataset
    trainData, testData = loadData(30, batchSize, dataset)

    # setting device as cpu or gpu
    device = setDevice()

    # creating new model or load model from directory
    model, criterion, optimizer = initModel(True, device, "", learningRate)

    # main training loop for model
    trainLosses, testLosses, trainAccuracies, testAccuracies = epochLoop(epochNumber, model, trainData, testData,
                                                                         criterion, device, optimizer, batchSize,
                                                                         learningRate)

    # SAVING NUMPY ARRAYS OF TRAIN and TEST RESULTS TO DRAW GRAPHS LATER, COMMENTED FOR NOW
    # np.save(f"./arrays/res_0.45_loss_test_batch-{batchSize}_lr-{learningRate}.npy", testLosses)
    # np.save(f"./arrays/res_0.45_loss_train_batch-{batchSize}_lr-{learningRate}.npy", trainLosses)
    # np.save(f"./arrays/res_0.45_accuracy_test_batch-{batchSize}_lr-{learningRate}.npy", testAccuracies)
    # np.save(f"./arrays/res_0.45_accuracy_train_batch-{batchSize}_lr-{learningRate}.npy", trainAccuracies)

    # Only applying test on model for dataset
    # testLoss, testAccuracy = testingModel(epochNumber, model, testData, criterion, device)
    # print(f"{testAccuracy * 100}%")

    # drawing graph for loss or accuracy array for specific model
    # trainAcc = np.load("./arrays/res_0.60_loss_train_batch-64_lr-0.0005.npy")
    # testAcc = np.load("./arrays/res_0.60_loss_test_batch-64_lr-0.0005.npy")
    # plotting(trainAcc, testAcc, "Loss", 64, 0.0005, "With Residual and 0.60 Dropout")


if __name__ == '__main__':
    main()
