from PIL import Image
import numpy as np
import glob
from keras.preprocessing.image import ImageDataGenerator
from multiprocessing.dummy import Pool as ThreadPool

def augment_images(raw_images, files, mult_factor):
    gen = ImageDataGenerator()
    for idx, image in enumerate(raw_images):
        for mult in range(mult_factor):
            img_fname = files[idx].split('/')[4]
            img_fname = '../../Data/AugmentedImages/' + \
                img_fname.split('.')[0] + '_' + str(multi) + '.jpg'

            theta_tfx = np.random.choice(range(270))
            transformed_raw_image = gen.apply_transform(image,
                                {'theta': theta_fx})
            new_image = Image.fromarray(transformed_raw_image, 'RGB')
            new_image = new_image.resize((1024, 1024), Image.ANTIALIAS)
            new_image.save(img_fname)
            transformed_raw_image = None
            new_image = None

if __name__ == '__main__':
    raw_images_dir = '../../Data/RawImages/'
    raw_image_files = sorted(glob.sglob(raw_images_dir + '*.jpg',
                                recursive=True))

    img_list = []
    for file in raw_image files:
        img_list.append(np.array(Image.open(file)))
    augment_images(img_list, raw_image_files, mult_factor=10)
