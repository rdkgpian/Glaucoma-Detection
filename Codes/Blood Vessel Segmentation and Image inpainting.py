
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

clahe=cv2.imread('drishtiGS_061clahe.png',0)
color=cv2.imread('drishtiGS_002.png',1)

crop_clahe = clahe[400:1400,500:1600]
crop_color_green=color[400:1400,500:1600,1:2]

def txn(img):
    low=np.min(img)
    high=np.max(img)
    a=np.empty(img.shape)
    a= ((img-low)/(high-low))*255
    return a


txn_clahe=txn(crop_clahe)
txn_green=txn(crop_color_green)


cv2.imwrite('txn_clahe.png',txn_clahe)
cv2.imwrite('txn_green.png',txn_green)

neg_clahe=255-txn_clahe
neg_gr=255-txn_green

cv2.imwrite('neg_clahe_1.png',neg_clahe)
cv2.imwrite('neg_green.png',neg_gr)

kernel = np.ones((5,5),np.uint8)
    
opening_clahe = cv2.morphologyEx(neg_clahe, cv2.MORPH_OPEN, kernel)
opening_gr=cv2.morphologyEx(neg_gr, cv2.MORPH_OPEN, kernel)

cv2.imwrite('opening_clahe_1.png',opening_clahe)
cv2.imwrite('opening_green.png',opening_gr)

os.chdir('/media/rd_kgpian/New Volume/Drishti_GS/')

def extract_bv(image):

    contrast_enhanced_green_fundus=image
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8 ))

# applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
    hsv = cv2.cvtColor(f4, cv2.COLOR_BGR2HSV)
    hsv_planes = cv2.split(hsv)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(50,50 ))
    hsv_planes[2] = clahe.apply(hsv_planes[2])
    hsv = cv2.merge(hsv_planes)
    f5=cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    f5=f5[:,:,1]
    
    # removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255
    im2=f6
    contours, hierarchy = cv2.findContours(im2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    #vessels and also in an interval of area
    fundus_eroded = cv2.bitwise_not(newfin)
    xmask = np.ones(fundus.shape[:2], dtype="uint8") * 255
    x1=fundus_eroded
    xcontours, xhierarchy = cv2.findContours(x1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)  
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)
    
    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)
    #blood_vessels = cv2.bitwise_not(finimage)
    return finimage
    #return contrast_enhanced_green_fundus

if __name__ == "__main__":
    pathFolder = "Drishti-GS1_files/Training/Clahe_images/"
    filesArray = [x for x in os.listdir(pathFolder) if os.path.isfile(os.path.join(pathFolder,x))]
    destinationFolder = "Drishti-GS1_files/Training/Blood_Vessel_Seg/"
    if not os.path.exists(destinationFolder):
        os.mkdir(destinationFolder)
    for file_name in filesArray:
        file_name_no_extension = os.path.splitext(file_name)[0]
        fundus = cv2.imread(pathFolder+'/'+file_name)
        #print(fundus)
        bloodvessel = extract_bv(fundus)
        #cv2.imwrite(destinationFolder+file_name_no_extension+"_bloodvessel.png",bloodvessel)
        cv2.imwrite(destinationFolder+file_name_no_extension+"_bloodvesel.png",bloodvessel)