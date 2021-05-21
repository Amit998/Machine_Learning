import cv2
import numpy as np

net=cv2.dnn.readNetFromTensorflow(
    "dnn/frozen_inference_graph_coco.pb",
    "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
    )

img=cv2.imread('road.jpg')

height,width,_=img.shape



colors=np.random.randint(0,255,(80,3))

print(colors)

#create black Image

black_image=np.zeros((height,width,3),np.uint8)

black_image[:]=(100,100,0)

#Detect Objects

blob=cv2.dnn.blobFromImage(img,swapRB=True)
net.setInput(blob)
boxes,masks=net.forward(["detection_out_final","detection_masks"])

# print(boxes)


# box=boxes[0,0,0]




detection_count=boxes.shape[2]
# print(detection_count)

# print(masks)

for i in range(detection_count):
    box=boxes[0,0,i]
    # print(box)
    class_=box[1]
    scores=box[2]

    if scores < 0.7:
        continue

    # get box coordinates
    x=int(box[3] * width)
    y=int(box[4] * height)
    x2=int(box[5] * width)
    y2=int(box[6] * height)

    roi=black_image[y:y2,x:x2]

    roi_hight,roi_width,_=roi.shape

    # Get The Mask

    mask=masks[i,int(class_)]

    mask=cv2.resize(mask,(roi_width,roi_hight))
    _,mask=cv2.threshold(mask,0.5,255,cv2.THRESH_BINARY)
    
    # print(mask.shape)

    # print(class_,scores)
    # cv2.rectangle(img,(x,y),(x2,y2),255,3)
    # cv2.imshow("Image",img)

    # cv2.imshow("mask",mask)
    # cv2.waitKey(0)

    color=colors[int(class_)]
    _,contours,_=cv2.findContours(np.array(mask,np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:

        cv2.fillPoly(roi, [cnt], (int(color[0]) ,int(color[1]) ,int(color[2])))
        # print(cnt)
        
    
    cv2.imshow('roi',roi)
    cv2.waitKey(10)


# print(box)

# x=int(box[3] * width)
# y=int(box[4] * height)
# x2=int(box[5] * width)
# y2=int(box[6] * height)

# print(x,y,x2,y2)

# cv2.rectangle(img,(x,y),(x2,y2),255,3)

cv2.imshow("image",img)
cv2.imshow("Black Image",black_image)
cv2.waitKey(0)