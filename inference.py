import argparse
from hyperparameters_dto import InferenceConfiguration
from config import Config
import cv2 as cv
# 
# : Implement the inference class

class Inference:

    inference_cfg: InferenceConfiguration=None

    # TODO: Develop better inference    
    def __init__(self, config: Config):
        cfg = config.config
        
        self.inference_cfg = InferenceConfiguration(cfg["parameters"])
    
    
    def read_image(self):
        self.img = cv.imread(self.inference_cfg.img_input, cv.IMREAD_COLOR)
    
    def preprocess(self):
        pass
    def load_model(self):
        pass
    
    def predict(self):
        pass
    
    def save_output(self):
        pass
    
    
def main(args: argparse.Namespace):
    cfg = Config(args.cfg)
    
    inference = Inference(cfg)
    inference.preprocess()
    inference.load_model()  
    inference.predict()
    inference.save_output()
    
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog='Custom CNN Inference',
        description='Custom CNN Inference')
    parser.add_argument("-c", "--cfg", default="./inference.yaml", required=False)
    
    args = parser.parse_args()
    
    main()    




