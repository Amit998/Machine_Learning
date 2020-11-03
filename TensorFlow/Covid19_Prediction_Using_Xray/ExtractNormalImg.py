import shutil
import os
import random
KAGGLE_FILE_PATH="chest_xray_kaggle/train/NORMAL"

TARGET_NORMAL_DIR="Dataset/Normal"


image_name=os.listdir(KAGGLE_FILE_PATH)

random.shuffle(image_name)
for i in range(142):
    image_name=image_name[i]
    image_path=os.path.join(TARGET_NORMAL_DIR,image_name)
    target_path=os.path.join(TARGET_NORMAL_DIR,image_name)
    shutil.copy2(image_path,target_path)
    print("copyig...",i)