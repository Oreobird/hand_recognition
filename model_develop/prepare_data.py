import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pickle
from sklearn.model_selection import train_test_split

def gen_background(data_dir):
    for i in range(2000):
        bg = np.random.randint(50, 200, (96, 96), dtype=np.uint8)
        img = Image.fromarray(bg)
        bg_dir = os.path.join(data_dir, '5_bg')
        os.makedirs(bg_dir, exist_ok=True)
        img.save(os.path.join(bg_dir, f'{i}.png'))

def process_images(imagepaths, process_func=None):
    X = []
    y = []
    num = 0
    for path in imagepaths:
        img = Image.open(path)
        img = img.resize((96, 96)).convert('L')
        if process_func:
            img = process_func(img)
        img_array = np.array(img)
        X.append(img_array.reshape(96, 96, 1))
        category = path.split("/")[-2]
        label = int(category.split("_")[0])
        y.append(label)
        print("{}-{}:{}".format(y[num], num, path))
        num += 1
    return X, y

def prepare_data(data_dir, save_crop=False):
    imagepaths = []
    for root, dirs, files in os.walk(data_dir, topdown=False):
        for name in files:
            path = os.path.join(root, name)
            if path.endswith("png"):
                imagepaths.append(path)
    print("image number:{}".format(len(imagepaths)))

    print("process orig")
    X, y = process_images(imagepaths)

    print("process rotate 45")
    X_rotate_45, y_rotate_45 = process_images(imagepaths, process_func=lambda img: img.rotate(45))
    X.extend(X_rotate_45)
    y.extend(y_rotate_45)

    print("process rotate -45")
    X_rotate_neg_45, y_rotate_neg_45 = process_images(imagepaths, process_func=lambda img: img.rotate(-45))
    X.extend(X_rotate_neg_45)
    y.extend(y_rotate_neg_45)

    print("process rotate brightness")
    X_brightness, y_brightness = process_images(imagepaths, process_func=lambda img: ImageEnhance.Brightness(img).enhance(0.7))
    X.extend(X_brightness)
    y.extend(y_brightness)

    print("process rotate blur")
    X_blur, y_blur = process_images(imagepaths, process_func=lambda img: img.filter(ImageFilter.GaussianBlur(0.8)))
    X.extend(X_blur)
    y.extend(y_blur)

    X = np.array(X, dtype="uint8")
    y = np.array(y)

    print("X number:{}, y number:{}".format(len(X), len(y)))

    ts = 0.3
    X_train, X_test1, y_train, y_test1 = train_test_split(X, y, test_size=ts, random_state=42)
    X_test, X_cal, y_test, y_cal = train_test_split(X_test1, y_test1, test_size=ts, random_state=42)

    with open(os.path.join(data_dir, '../X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(data_dir, '../y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(data_dir, '../X_test.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(data_dir, '../y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    with open(os.path.join(data_dir, '../X_cal.pkl'), 'wb') as f:
        pickle.dump(X_cal, f)
    with open(os.path.join(data_dir, '../y_cal.pkl'), 'wb') as f:
        pickle.dump(y_cal, f)

if __name__ == "__main__":
    work_dir = '/content/drive/MyDrive/hand_recognition/data/leapGestRecog'
    # gen_background(work_dir)
    prepare_data(work_dir)

