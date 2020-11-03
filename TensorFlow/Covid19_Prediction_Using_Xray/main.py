import pandas as pd
import os
import shutil


FILE_PATH="chestxray/metadata.csv"
IMAGE_PATH="chestxray/images"

df=pd.read_csv(FILE_PATH)
# print(df.head(20))

TARGET_DIR="dataset/Covid"


if not os.path.exists(TARGET_DIR):
    os.mkdir(TARGET_DIR)
    print("Covid Folder created")

cnt=0
total=0

for (i,row) in df.iterrows():
    if(row["finding"]=="Pneumonia/Viral/COVID-19" and row["view"]=="PA" ):
        filename=row["filename"]
        image_path=os.path.join(IMAGE_PATH,filename)
        image_copy_path=os.path.join(TARGET_DIR,filename)
        shutil.copy2(image_path,image_copy_path)
        print(("Moving Image",cnt))
        cnt+=1
    total+=1


    
print(total)
print(cnt)