import os

import numpy as np


def write_data_in_numpy(data, path, file_name = "data.npy"):
    file_path = os.path.join(path, file_name)
    np.save(file_path, data)

if __name__ == "__main__":
    data = [{'name': 'yuanshen'}]
    write_data_in_numpy(data, "D:/Study/Projects/DeepLearning/UrbanGeneration/Indoor/ATISS Remake/")
