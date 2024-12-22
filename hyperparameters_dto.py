
from dataclasses import dataclass, fields


class ConfigurationClass:
    def __init__(self, cfg: dict):
        for field, value in zip(fields(self), cfg.values()):
            setattr(self, field.name, value)


@dataclass
class TrainConfiguration(ConfigurationClass):
    """ Train hyperparameters dto"""
    device: str
    learning_rate: float
    epochs: int
    batch_size: int
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        
        
@dataclass    
class DatasetConfiguration(ConfigurationClass):
    """ Dataset Configuration dto"""
    dataset_directory: str
    train_set_length: int
    test_set_length: int
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)

@dataclass    
class LoggingConfiguration(ConfigurationClass):
    """ Logging Configuration dto"""
    directory: str 
    sub_directory: str
    model_name: str
        
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
    
@dataclass    
class ModelSaveConfiguration(ConfigurationClass):
    """ Model Sacing Configuration dto"""
    save_directory: str
    name: str
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)