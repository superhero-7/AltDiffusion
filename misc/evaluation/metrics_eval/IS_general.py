from geval.image_dataset import ImageDataset 
from geval.inception_score import inception_score

ds = ImageDataset("/home/xingzhaohu/sharefs/datasets/car_segmentation/train", image_type="jpg")

out = inception_score(ds, resize=True, batch_size=32, splits=2)

print(out)