import torch, io
from PIL import Image
from typing import List, Dict, Any

## weights_ = None, if weights_: custom, else: pretrained

class Yolo:

    def __init__(self, weights_: str, iou: int = 0.45, 
                    conf: int = 0.70, classes: List[int] = [2,3]) -> None:
        torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_, force_reload=True)
        self.model.eval() # using only for inference so set to eval mode
        self.model.iou = iou
        self.model.conf = conf
        self.model.classes = classes # 2: pool, 3: solar
        self._files = None

    @property
    def files(self):
        """ 
        'files' property, getter method
        """
        return self._files

    @files.setter
    def files(self, value):
        """ 
        setter method
        """
        self._files = value

    @files.deleter
    def files(self):
        """ 
        deleter method
        """
        del self._files

    def process_data(self) -> List[Any]:
        imgs = []
        for file in self.files:
            img_bytes = file.read()
            img_bw = Image.open(io.BytesIO(img_bytes))
            img = img_bw.convert('RGB')
            imgs.append(img)
        return imgs

    def get_predictions(self, size: int = 416) -> Dict[int, Any]:
        imgs = self.process_data()
        results = self.model(imgs, size=size)
        predictions = results.pandas().xyxy # list of predictions for each img
        data = {
            i: predictions[i].to_dict(orient='records') for i in range(len(predictions))
            }
        return data