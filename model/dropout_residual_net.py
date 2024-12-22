from torch import nn

class DropoutResidualNet(nn.Module):
    def __init__(self):
        super().__init__()

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

        # resize as a vector of (batch_size, -1) -> (flatten)
        out = out.view(out.size(0), -1)

        # fc1
        out = self.drop1(out)
        out = self.fc1(out)
        out = self.relu6(out)

        # fc2
        out = self.drop2(out)
        out = self.fc2(out)
        return out