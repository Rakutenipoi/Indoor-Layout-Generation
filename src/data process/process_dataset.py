import json
import os
from read_dataset import read_folder_files_json








if __name__ == "__main__":
    json_folder_path = "D:/Study/Projects/DeepLearning/Resources/Indoor/3D-FRONT/"

    data = read_folder_files_json(json_folder_path, 10)

    print("done")
