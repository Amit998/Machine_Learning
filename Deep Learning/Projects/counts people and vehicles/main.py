import cv2
from vehicle_detector import VehicleDetector
import glob


vd=VehicleDetector()


image_folder=glob.glob("images/*.*")

vehicle_folder_count=0


for image_path in image_folder:
    print("Img", image_path)
    img=cv2.imread(image_path)


    vehicle_boxes=vd.detect_vehicles(img)
    vehicle_count=len(vehicle_boxes)

    vehicle_folder_count += vehicle_count
    print("Total current Count",vehicle_folder_count)


    for box in vehicle_boxes:
        x,y,w,h=box

        cv2.rectangle(img, (x,y),(x+w,y+h),(25,0,180),3)
        cv2.putText(img,"Vechile: "+str(vehicle_count),(20,50),0,2,(100,200,0),3)


    # print(vehicle_boxes)

    cv2.imshow("Cars",img)

    cv2.waitKey(1)
