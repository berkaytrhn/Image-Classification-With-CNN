import argparse
import os
from time import time
import math

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from tqdm import tqdm

from config import Config
from hyperparameters_dto import DatasetConfiguration, TrainConfiguration, ModelSaveConfiguration
from model import BaseNet, DropoutNet, DropoutResidualNet, ResidualNet

class Train:
    
    # Configurations
    data_cfg: DatasetConfiguration=None
    train_cfg: TrainConfiguration=None
    model_cfg: ModelSaveConfiguration=None


    def __init__(self, config: Config):
        cfg = config.config
        self.data_cfg = DatasetConfiguration(cfg["data"])
        self.train_cfg = TrainConfiguration(cfg["train"])
        self.model_cfg = ModelSaveConfiguration(cfg["model"])
        
    #TODO: implement the trainer class methods
    
    
    def set_device(self) ->None:
        # set device
        if torch.cuda.is_available() and self.train_cfg.device == "cuda":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    def load_dataset(self) -> None:
        
        # transforms for training process
        transform = transforms.Compose([
            # augmentation
            transforms.RandomHorizontalFlip(p=0.3), 
            transforms.RandomRotation(degrees=40),
            transforms.CenterCrop(128),

            # resize and normalization(-1, 1)
            transforms.Resize(128),
            transforms.ToTensor(),
            # expect relu to work better when data is zero centered
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
        # dataset from directory
        dataset = ImageFolder(self.data_cfg.dataset_directory, transform=transform) 
    
        # train test set separation 
        dataset_length = len(dataset)
        trainNumber = math.floor(self.data_cfg.train_set_length * dataset_length)
        testNumber = math.ceil(self.data_cfg.test_set_length  * dataset_length)
        trainSet, testSet = random_split(
            dataset, 
            (trainNumber, testNumber), 
            generator=torch.Generator().manual_seed(42)
        )
        
        # initializing data loaders
        self.trainData = DataLoader(
            dataset=trainSet,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            num_workers=4
        )

        self.testData = DataLoader(
            dataset=testSet,
            batch_size=self.train_cfg.batch_size,
            shuffle=False,
            num_workers=4
        )
        
    def init_model(self) -> None:
        # initializing model
        if self.train_cfg.model == "BaseNet":
            self.model = BaseNet().to(self.device)
        elif self.train_cfg.model == "DropoutNet":
            self.model = DropoutNet().to(self.device)
        elif self.train_cfg.model == "DropoutResidualNet":
            self.model = DropoutResidualNet().to(self.device)
        elif self.train_cfg.model == "ResidualNet":
            self.model = ResidualNet().to(self.device)

    def save_model(self):
        torch.save(
            self.model, 
            os.path.join(
                self.model_cfg.save_directory, 
                f"{self.model_cfg.name}_{self.train_cfg.epochs}.pth"
            )
        )

    def configure_training(self) -> None:
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.train_cfg.learning_rate
        )
        # TODO: try torch.compile
        # model = torch.compile(UNet().to(device))
    
    def _training_step(self, epoch: int) -> None:
        # method for training model
        totalLoss = 0
        start = time()

        accuracy = []

        progress_bar = tqdm(self.trainData, desc="Training...")


        for i, (X, y) in enumerate(progress_bar):

            X = X.to(self.device)
            y = y.to(self.device)

            # output of model
            out = self.model(X)

            # computing the cross entropy loss
            loss = self.criterion(out, y)
            totalLoss += loss.item()

            # zeroing the gradients
            self.optimizer.zero_grad()

            # Back propogation through loss
            loss.backward()

            # updating model parameters
            self.optimizer.step()

            argmax = out.argmax(dim=1)
            # calculating accuracy by comparing to target
            accuracy.append((y == argmax).sum().item() / self.train_cfg.batch_size)

            if i % self.train_cfg.print_every == 0:
                # TODO: Loss starts from 0 and acc from 1, fix bugs
                desc = f"Epoch: {epoch}, Train Loss: {loss.item():.4f}, Accuracy: {sum(accuracy) / len(accuracy):.2f}, Time: {time() - start:.2f}"
                progress_bar.set_description(desc)
            
            progress_bar.update(1)
        # Returning Average Training Loss and Accuracy
        return totalLoss, sum(accuracy) / len(accuracy)
    
    @torch.no_grad()
    def _validation_step(self):
        totalLoss = 0
        accuracy = []

        for i, (X, y) in enumerate(self.testData):
            
            
            X = X.to(self.device)
            y = y.to(self.device)

            # output by our model
            out = self.model(X)

            # computing the cross entropy loss
            loss = self.criterion(out, y)
            totalLoss += loss.item()

            argmax = out.argmax(dim=1)
            
            # Find the accuracy of the batch by comparing it with actual targets
            accuracy.append((y == argmax).sum().item() / self.train_cfg.batch_size)
            
        # Returning Average Testing Loss and Accuracy
        return totalLoss, sum(accuracy) / len(accuracy)
    
    def train(self):
        
        
        test_data_length = len(self.testData)
        train_data_length = len(self.trainData)
        
        # main loop
        trainLosses = []
        validationLosses = []
        trainAccuracies = []
        validationAccuracies = []

        for epoch in tqdm(range(1, self.train_cfg.epochs + 1)):
            trainLoss, trainAccuracy = self._training_step(epoch)
            validationLoss, validationAccuracy = self._validation_step()

            # train and validation losses calculated with precalculated data lengths
            trainLoss /= train_data_length
            validationLoss /= test_data_length



            trainLosses.append(trainLoss)
            validationLosses.append(validationLoss)
            trainAccuracies.append(trainAccuracy)
            validationAccuracies.append(validationAccuracy)
            




def main(args: argparse.Namespace):
    
    cfg = Config(args.cfg)
    
    trainer = Train(cfg)
    trainer.set_device()
    trainer.load_dataset()
    trainer.init_model()
    trainer.configure_training()
    trainer.train()
    trainer.save_model()
    
    


if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(
        prog='Custom CNN Train',
        description='Custom CNN Training Process')
    
    
    parser.add_argument("-c", "--cfg", default="./train.yaml", required=False)
    
    args = parser.parse_args()
    main(args)