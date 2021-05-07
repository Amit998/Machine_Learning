import cv2
import numpy as np
import os



path='image'
img2=cv2.imread("image/4.png")
###IMPORT IMAGES

images=[]
classNames=[]
myList=os.listdir(path)
# print(myList)


orb=cv2.ORB_create(nfeatures=1000)
print('Total classes detected:',len(myList))


for cl in myList:
    curImage=cv2.imread(f'{path}/{cl}')
    images.append(curImage)
    classNames.append(os.path.splitext(cl)[0])

# print(classNames)


def findDes(images):
    desList=[]
    for img in images:
        kp,des=orb.detectAndCompute(img,None)
        desList.append(des)
    
    return desList



def find_id(img,desList,threshold=15):
    kp2,des2=orb.detectAndCompute(img,None)
    bf=cv2.BFMatcher()   
    matchList=[]
    final_value=-1
    try:
        for des in desList: 
            matches=bf.knnMatch(des,des2,k=2)
            good=[]
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            
            matchList.append(len(good))
    except:
        pass

    if len(matchList)!=0:
        if max(matchList) > threshold:
            final_value=matchList.index(max(matchList))
    
    return final_value
    
    # print(matchList)



desList=findDes(images)


cap=cv2.VideoCapture(0)

while True:
    success,img2=cap.read()
    imgOriginal=img2.copy()
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)


    id=find_id(img2,desList)

    if id != -1:
        cv2.putText(imgOriginal,classNames[id],(50,50),cv2.FONT_HERSHEY_TRIPLEX,1,(200,20,21))
    

    cv2.imshow("Image ",imgOriginal)
    cv2.waitKey(1)



# print(desList)

# img1=cv2.imread("image/2.jpg")
# img2=cv2.imread("image/4.png")


# orb=cv2.ORB_create(nfeatures=1000)

# kp1,des1=orb.detectAndCompute(img1,None)
# kp2,des2=orb.detectAndCompute(img2,None)



# imgKp1=cv2.drawKeypoints(img1,kp1,None)
# imgKp2=cv2.drawKeypoints(img2,kp2,None)

# # print(des1,des2)

# bf=cv2.BFMatcher()
# matches=bf.knnMatch(des1,des2,k=2)


# good=[]

# for m,n in matches:
#     if m.distance < 0.75*n.distance:
#         good.append([m])


# print(len(good))
# img3=cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
# # cv2.imshow("kp1", imgKp1)
# # cv2.imshow("kp2", imgKp2)


# cv2.imshow("img1",img1)
# cv2.imshow("img2",img2)
# cv2.imshow("img3",img3)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()