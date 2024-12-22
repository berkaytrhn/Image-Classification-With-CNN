from config import Config
from torchvision.transforms import transforms
from hyperparameters_dto import DatasetConfiguration, TrainConfiguration, LoggingConfiguration, ModelSaveConfiguration  
import argparse


class Train:
    
    # Configurations
    data_cfg: DatasetConfiguration=None
    train_cfg: TrainConfiguration=None
    logging_cfg: LoggingConfiguration=None
    model_cfg: ModelSaveConfiguration=None


    def __init__(self, config: Config):
        cfg = config.config
        self.data_cfg = DatasetConfiguration(cfg["data"])
        self.train_cfg = TrainConfiguration(cfg["train"])
        self.logging_cfg = LoggingConfiguration(cfg["logging"])
        self.model_cfg = ModelSaveConfiguration(cfg["model"])
        
    #TODO: implement the trainer class methods


    def train(self):
        pass
    
    def test(self):
        pass




def main(args: argparse.Namespace):
    
    cfg = Config(args.cfg)
    
    trainer = Train(cfg)
    print(trainer)
    
    


if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(
        prog='Custom CNN Train',
        description='Custom CNN Training Process')
    
    
    parser.add_argument("-c", "--cfg", default="./config.yml", required=False)
    
    args = parser.parse_args()
    main(args)