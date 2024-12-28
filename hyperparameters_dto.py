
from dataclasses import dataclass, fields


class ConfigurationClass:
    def __init__(self, cfg: dict):
        for field, value in zip(fields(self), cfg.values()):
            setattr(self, field.name, value)


@dataclass
class TrainConfiguration(ConfigurationClass):
    """ Train hyperparameters dto"""
    model: str
    device: str
    learning_rate: float
    epochs: int
    batch_size: int
    print_every: int
    
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
class ModelSaveConfiguration(ConfigurationClass):
    """ Model Sacing Configuration dto"""
    save_directory: str
    name: str
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        
        
@dataclass
class InferenceConfiguration(ConfigurationClass):
    """ Inference Parameters dto"""
    img_input: str
    img_output: str
    model: str
    
    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)