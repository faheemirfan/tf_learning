from PIL import Image
import numpy as np
import os
from pathlib import Path
import glob
import sys
import random
from sklearn.model_selection import train_test_split


from loguru import logger
logger.remove()
logger.add(sys.stderr,level="DEBUG")



MAIN_FOLDER_PATH="/home/faheem/Learning/Tensorflow/repo/tf_learning/fourth_week/EuroSAT_RGB"
IMAGE_EXTENSIONS=['.jpg','.png']


class create_image_dataset():
    def __init__(self,image_folder):
        self.image_dataset = []
        self.image_labels = []
        self.images_folder = image_folder

    def read_image_and_convert_to_array(self,image_path):
        img = Image.open(image_path)
        img_array = np.asanyarray(img)/255.0
        return img_array


    def read_all_images_in_folder(self,folder_path,lab):
        images = []
        labels = []
        image_files = folder_path.glob('*')
        for image_path in image_files:
            if(not os.path.splitext(image_path)[1] in IMAGE_EXTENSIONS):
                logger.warning(f'{image_path} is not image file with required extension')
                continue
            images.append(self.read_image_and_convert_to_array(image_path))
            labels.append(lab)
        return images,labels
    

    def create_dataset(self):
        images_data = []
        labels = []
        list_folders = sorted(os.listdir(self.images_folder))
        for i,folder_name in enumerate(list_folders):
            folder_path = Path(MAIN_FOLDER_PATH,folder_name)
            read_images,read_labels = self.read_all_images_in_folder(folder_path,i)
            images_data.extend(read_images)
            labels.extend(read_labels)

        combine_images_labels = list(zip(images_data,labels))
        random.shuffle(combine_images_labels)
        images_data, labels = zip(*combine_images_labels)
        self.image_dataset = np.stack(images_data)
        self.image_labels = np.stack(labels)

    def load_dataset(self,testSize=0.1 ):
        self.create_dataset()
        train_data,test_data,train_labels,test_labels = train_test_split(self.image_dataset,
                                                                         self.image_labels,
                                                                         test_size=testSize,
                                                                         random_state=42)
        train_labels = train_labels.astype(np.uint8)
        test_labels = test_labels.astype(np.uint8)
        
        return train_data,test_data,train_labels,test_labels


if __name__=='__main__':
    image_read_obj = create_image_dataset(MAIN_FOLDER_PATH)
    image_read_obj.load_dataset()