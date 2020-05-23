import pandas as pd
import os
from torchvision import transforms
from PIL import Image

class ImageLoader():
    """To load images from image_folder for model training

    Arguments :

    image_folder : Image folder path
    label_csv : CSV file with image name and text of the image
    gray_scale : whether to load the image in b&w or in color
    transform : transformation to be applied on loaded image"""   

    def __init__(self, image_folder, label_csv, transform = None, gray_scale = True) :
        """assign values to the attributes"""   
        self.image_folder = image_folder
        self.label_csv = pd.read_csv(label_csv)
        self.transform = transform
        self.gray_scale = gray_scale

    def __len__(self):
        """Returns length of CSV"""
        return len(self.label_csv)
    
    def __getitem__(self, index) :
        """Returns specific row data from labeled csv after transformation"""  

        assert isinstance (index, int)    
        assert index < len(self)
        image = os.path.join(self.image_folder,self.label_csv['Image_Name'][index])
        label = self.label_csv['Text'][index]
        label = label.strip()
        image = Image.open(image)

        if self.gray_scale :
            if image.mode != 'L':
                image = image.convert('L')

        if self.transform != None :
            image = self.transform(image)

        sample = {'image':image,'label':label}
        
        return sample  


