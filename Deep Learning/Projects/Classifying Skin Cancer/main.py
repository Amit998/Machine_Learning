import os
import shutil
import random

seed =1
random.seed(seed)
directory="D:/study/datasets/Skin Cancer/"



train="data/train/"
test="data/test/"

validation="data/validation/"

os.makedirs(train+"bengins/")
os.makedirs(train+"malignant/")

os.makedirs(test+"bengins/")
os.makedirs(test+"malignant/")

os.makedirs(validation+"bengins/")
os.makedirs(validation+"malignant/")


test_examples=train_examples=validation_examples=0
counter=1

for line in open(f"{directory}/ISIC_2019_Training_GroundTruth.csv").readlines()[1:]:
    print(counter)
    split_line=line.split(",")
    img_file=split_line[0]
    benign_malin=split_line[1]

    # print(benign_malin)
    random_num=random.random()

    if (random_num < 0.8):
        location = train
        train_examples+=1
    elif random_num < 0.9:
        location = validation
        validation_examples+=1
    else:
        location = test
        test_examples+=1
    
    # print(location+img_file+".jpg")
    
    if (int(float(benign_malin))==0):
        shutil.copy(
            "D:/study/datasets/Skin Cancer/ISIC_2019_Training_Input/ISIC_2019_Training_Input/"+img_file+".jpg",
            location+"bengins/"+img_file+".jpg",
        )
    elif (int(float(benign_malin))==1):
        shutil.copy(
            "D:/study/datasets/Skin Cancer/ISIC_2019_Training_Input/ISIC_2019_Training_Input/"+img_file+".jpg",
            location+"malignant/"+img_file+".jpg",
        )
    counter+=1

print(f"number of traning examples {train_examples}")
print(f"number of test examples {test_examples}")
print(f"number of validation examples {validation_examples}")