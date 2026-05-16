import numpy as np
import PIL.Image as Image
import os
from tensorflow.keras.applications.inception_v3 import InceptionV3

def image_loader(filepath, dimensions=(32, 32)):
    # load gambar dari file dengan PIL
    img = Image.open(filepath)
    img = img.convert('RGB')
    img = img.resize(dimensions) 
    img_array = np.array(img, dtype=np.float32)
    img_array /= 255.0  # normalisasi ke [0, 1]

    return img_array


def batch_loader(filepaths, dimensions=(32, 32)):
    # load dan memproses sekumpulan gambar jadi numpy array (N, H, W, C)
    images = []
    for filepath in filepaths:
        img_array = image_loader(filepath, dimensions)
        images.append(img_array)

    return np.array(images)

def feature_extractor(filepaths, output_filename="features.npy", model=None):
    # menerima list path, menggunakan keras CNN encoder untuk ekstraksi fitur, dan menyimpan hasil ke disk format .npy
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    for layer in model.layers:
        layer.trainable = False
        
    features_dict = {}
    for filepath in filepaths:
        img_array = image_loader(filepath, dimensions=(299, 299))  # InceptionV3 expects 299x299
        if model is not None:
            features = model.predict(np.expand_dims(img_array, axis=0))[0]

        features_dict[filepath] = features
    np.save('src/weights/' + output_filename, features_dict)
    
    print(f"Fitur berhasil disimpan ke {output_filename}")


# CARA PAKAI
if __name__ == "__main__":
    # folder dataset
    folder_gambar = "data/cnn/Images"
    
    list_path = [os.path.join(folder_gambar, f) for f in os.listdir(folder_gambar) if f.endswith('.jpg') or f.endswith('.png')]
    
    # Panggil methodnya
    feature_extractor(list_path, "bobot_ekstraksi_flickr8k.npy")