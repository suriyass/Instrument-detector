import cv2
import numpy as np
from google.colab import files
uploaded = files.upload()
from google.colab.patches import cv2_imshow
img = cv2.imread('i.jpg', 0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

height,width = img.shape
mask = np.zeros((height,width), np.uint8)

counter = 0

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=200,param2=100,minRadius=0,maxRadius=0)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)

    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

    # Draw on mask
    cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),-1)
    masked_data = cv2.bitwise_and(cimg, cimg, mask=mask)    

    # Apply Threshold
    _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

    # Find Contour
    cnt = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

    #print len(contours)
    x,y,w,h = cv2.boundingRect(cnt[0])

    # Crop masked_data
    crop = masked_data[y:y+h,x:x+w]

    # Write Files
    cv2.imwrite("output/crop"+str(counter)+".jpg", crop)

    counter +=1

print (counter)
#from google.colab.patches import cv2_imshow
cv2_imshow(cimg)
cv2.imwrite("output/circled_img.jpg", cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
