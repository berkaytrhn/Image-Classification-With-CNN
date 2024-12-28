import argparse
from hyperparameters_dto import InferenceConfiguration
from config import Config
import cv2 as cv
from torchvision.transforms import transforms
import torch

class Inference:

    mappings = {
        0: 'airport_inside', 
        1: 'artstudio', 
        2: 'bakery', 
        3: 'bar', 
        4: 'bathroom', 
        5: 'bedroom', 
        6: 'bookstore', 
        7: 'bowling', 
        8: 'buffet', 
        9: 'casino', 
        10: 'church_inside', 
        11: 'classroom', 
        12: 'closet', 
        13: 'clothingstore',
        14: 'computerroom'
    }

    inference_cfg: InferenceConfiguration=None

    # TODO: Develop better inference    
    def __init__(self, config: Config):
        cfg = config.config
        
        self.inference_cfg = InferenceConfiguration(cfg["parameters"])
    
    
    def read_image(self):
        self.img = cv.imread(self.inference_cfg.img_input, cv.IMREAD_COLOR)
    
    def preprocess(self):
        _transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(128),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        processed_image = _transforms(self.img)
        self.processed_image = torch.unsqueeze(processed_image, 0).to(self.inference_cfg.device)
        
    def load_model(self):
        self.model: torch.nn.Module = torch.load(self.inference_cfg.model).to(self.inference_cfg.device)
        
    def predict(self):
        output = self.model(self.processed_image)
        index_max = output.argmax(dim=1)
        _class = self.mappings[index_max.item()]
        print(f"Predicted class: {_class}")
        self.post_process(_class)
        
    def post_process(self, text):
        self.output_img =  cv.putText(
            self.img,
            text,
            (30,30),
            cv.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    
    def save_output(self):
        cv.imwrite(self.inference_cfg.img_output, self.output_img)
    
    
def main(args: argparse.Namespace):
    cfg = Config(args.cfg)
    
    inference = Inference(cfg)
    inference.read_image()
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
    
    main(args)    




