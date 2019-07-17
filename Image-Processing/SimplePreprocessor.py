import cv2

class SimplePreprocessor:
    def __init__(self, wid, hei, inter=cv2.INTER_AREA):
        self.wid = wid
        self.hei = hei
        self.inter = inter
    
    def preporcess(self, image):
        return cv2.resize(image, (self.wid, self.hei), interpolation=self.inter)