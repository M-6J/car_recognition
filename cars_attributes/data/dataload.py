import os

def download_dataset(dataset_name, save_path):
    os.system(f"!pip install kaggle")
    os.system(f"!cp ./car_recognition/kaggle.json ~/.kaggle/")
    os.system(f'!chmod 600 ~/.kaggle/kaggle.json')
    os.system(f"!kaggle datasets download -d m6789j/{dataset_name}")
    os.system(f"!unzip cars196.zip -d {save_path}")
    