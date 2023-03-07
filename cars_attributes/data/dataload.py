import os
import json
import cv2

def download_dataset(dataset_name, save_path):
    # Check if dataset is already downloaded
    if os.path.exists(save_path):
        print(f"{dataset_name} dataset already exists at {save_path}.")
        return
    else:
    # Download the dataset metadata
        os.system(f"pip install kaggle")
        os.system(f"cp ./car_recognition/kaggle.json ~/.kaggle/")
        os.system(f'chmod 600 ~/.kaggle/kaggle.json')
        os.system(f"kaggle datasets download -d m6789j/{dataset_name}")
        os.system(f"unzip {dataset_name}.zip -d {save_path}")

        with open("./car_recognition/cars_annos/new_train.json", encoding='utf-8') as f:
            train_data = json.load(f)
            print("train_images:",len(train_data))
        with open("./car_recognition/cars_annos/new_test.json", encoding='utf-8') as f:
            test_data = json.load(f)
            print("val_images:",len(test_data))

        #Crop images with bounding box
        for i in range(len(train_data)):
            img = cv2.imread('./car_recognition/dataset/cars_train/images/' + train_data[i]['image_path'])
            x1 = train_data[i]['bbox_x1']
            y1 = train_data[i]['bbox_y1']
            x2 = train_data[i]['bbox_x2']
            y2 = train_data[i]['bbox_y2']
            output_img = img[y1:y2, x1:x2]
            cv2.imwrite('./car_recognition/dataset/cars_train/images/' + train_data[i]['image_path'], output_img)

        for i in range(len(test_data)):
            img = cv2.imread('./car_recognition/dataset/cars_test/images/' + test_data[i]['image_path'])
            x1 = test_data[i]['bbox_x1']
            y1 = test_data[i]['bbox_y1']
            x2 = test_data[i]['bbox_x2']
            y2 = test_data[i]['bbox_y2']
            output_img = img[y1:y2, x1:x2]
            cv2.imwrite('./car_recognition/dataset/cars_test/images/' + test_data[i]['image_path'], output_img)

        print(f"Successfully downloaded dataset {dataset_name} to {save_path}.")