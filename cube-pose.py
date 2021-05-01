# section 1 : extract the colored cube .
import cv2
import numpy as np
import os ,sys
import matplotlib.pyplot as plt

# calibration matrices
cmat = np.float32([[1590, 0, 884], [0, 1610, 639], [0, 0, 1]])
dpr = np.float32([0.207, 0.724, 0.0211, -0.0177, 0.682])

# parameter arrays
# threshold value array
hsv_value = np.empty([6], dtype=int)
#shadow mask removal parameter
sht = np.empty([6], dtype=int)
# histoghram displacement parameter
dispcmnt = np.empty([2], dtype=np.uint8)

## rgb threshold parameter
hsv_value[0] =  0
hsv_value[2] = 109
hsv_value[4] = 140
hsv_value[1] =  6
hsv_value[3] =  255
hsv_value[5] =  255
##
sht[0] =  100
sht[2] = 100
sht[4] = 150
sht[1] =  230
sht[3] =  230
sht[5] =  220
###
dispcmnt[0] = 25
#adaptive histoghram parameter
adpv_grids = 60
#minimal distance between tow edge point of cube shape
minDistance = 110
mc = 2500
#
# hitoghram offset function : to make histoghram more efficient ( pre process)
def histrans(frame_u):
    # create array at same frame size
    sub = np.full(frame_u.shape, dispcmnt[0])
    #subtract frames
    frame_i = cv2.subtract(frame_u, sub)
    return (frame_i)

#make adaptive histoghram at LAB color plane to eliminate light variation :
def adpv_hsm(frame_p):
    lab = cv2.cvtColor(frame_p, cv2.COLOR_BGR2LAB)
    planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(adpv_grids, adpv_grids))
    # applay the step only on b plane (brightness scale)
    planes[2] = clahe.apply(planes[2])
    frame_out1 = cv2.merge(planes)
    frame_out1 = cv2.cvtColor(frame_out1, cv2.COLOR_LAB2BGR)
    return (frame_out1)

def adjust_gamma(image, gamma=0.85):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

#calibrate the raw image taken from camera to correct the distortion .
def calbration(frame_c):
    h, w = frame_c.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cmat, dpr, (w, h), 1, (w, h))
    undist = cv2.undistort(frame_c, cmat, dpr, None, newcameramtx)
    x, y, w, h = roi
    frame_correct = undist[y:y + h, x:x + w]
    return frame_correct

# filter the frame to extract the specific color of cube .
def clr_filt(frame_a, orginal):
    global  minDistance
    hsv = cv2.cvtColor(frame_a, cv2.COLOR_BGR2HSV)
    hsvp = saveid + "\\" + "hsv" + img
    #cv2.imwrite(hsvp, hsv)
    lower_red = np.array([hsv_value[0], hsv_value[2], hsv_value[4]])
    upper_red = np.array([hsv_value[1], hsv_value[3], hsv_value[5]])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    # proccesing
    kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # IT WAS RECT
    mask2 = openning_morph(mask)

    res1 = cv2.bitwise_and(orginal, orginal, mask=mask2)
    res2 = shdmsk(res1)
    res = cv2.subtract(res1, res2)
    # inverse black to white
    n, w, h = res.shape
    for i in range(n):
        for j in range(w):
            g = res[i, j]
            if (g[0] == 0 and g[1] == 0 and g[2] == 0):
                res[i, j] = [255, 255, 255]
    filtred = res
    #cv2.imshow('filtered', filtred)
    ## save mask and filtred frame
    mpath = saveid + "\\" + "MASK" + img
    fpath = saveid + "\\" + "filtred" + img
    cv2.imwrite(mpath,mask)
    cv2.imwrite(fpath, filtred)
    # calculate histoghram for calculating object2im_ratio
    hist = cv2.calcHist([mask2], [0], None, [256], [0, 256])
    object2im_ratio = (hist[255]/hist[0])
    minDistance = mc * object2im_ratio
    return (filtred)
#shadow removal function
def shdmsk(filtredo): ## shadow mask
    lower_s = np.array([sht[0], sht[2], sht[4]])
    upper_s = np.array([sht[1], sht[3], sht[5]])
    mask = cv2.inRange(filtredo, lower_s, upper_s)
    unshad = cv2.bitwise_and(filtredo, filtredo, mask=mask)
    return (unshad)
def openning_morph(e):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)) # 3,3
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernal = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))  # IT WAS RECT
    e = cv2.dilate(e, kernal, iterations=2)
    #e = cv2.erode(e, kernal, iterations=5)
    e1 = mfil(e)
    e3 = cv2.morphologyEx(e1, cv2.MORPH_CLOSE, kernel2, iterations=5)
    e4 = cv2.morphologyEx(e3, cv2.MORPH_OPEN, kernel1, iterations=2)  # 2
    return (e4)
#filling mask
def mfil(masked):
    # Copy the thresholded image.
    im_floodfill = masked.copy()
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = masked.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255);
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    # Combine the two images to get the foreground.
    im_out = masked | im_floodfill_inv
    return im_out

def gc2trk(frame):  # gray scaled image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray1 = np.float32(gray)
    corner = cv2.cornerHarris(gray, 13, 11, 0.1)
    cl, rl = gray.shape
    print (minDistance)
    cornrz = cv2.goodFeaturesToTrack(gray, 6, 0.0001, 110,
                                     corner)  # ,mask,7,7,7) #maxCorners ,qualityLevel ,minDistance , op mask ,
    cornrz1 = np.reshape(cornrz,(6,2))
    cornrz = np.int0(cornrz)
    for i in cornrz:
        x, y = i.ravel()
        #print (x, y)
        cv2.circle(frame, (x, y), 3,(0,255,0), -1)
    #cv2.imshow(" gftd", frame)
    #tt = spath
    # cv2.imwrite(tt,frame)
    return (cornrz1)


def sortpoint(xy):
    global sorm
    sorm = xy
    xy=np.array(xy)
    x = np.array([[xy[0, 0]], [xy[1, 0]], [xy[2, 0]], [xy[3, 0]], [xy[4, 0]], [xy[5, 0]]])
    y = np.array([[xy[0, 1]], [xy[1, 1]], [xy[2, 1]], [xy[3, 1]], [xy[4, 1]], [xy[5, 1]]])
    x, y = sor2elment(x, y, 1, 2)
    x, y = sor2elment(x, y, 0, 3)
    x, y = sor2elment(x, y, 5, 4)
    # print (sorm)
    return (sorm)

def sor2elment(x, y, t1, t2):
    global sorm
    ds1 = np.empty([2, 2], dtype="double")
    e1 = np.argmax(x)
    ds1[0, 0] = x[e1]
    ds1[0, 1] = y[e1]
    x = np.delete(x, e1)
    y = np.delete(y, e1)
    e1 = np.argmax(x)
    ds1[1, 0] = x[e1]
    ds1[1, 1] = y[e1]
    x = np.delete(x, e1)
    y = np.delete(y, e1)
    e1 = np.argmax(ds1[:, 1])
    e2 = np.argmin(ds1[:, 1])
    sorm[t1, :] = ds1[e1, :]
    sorm[t2, :] = ds1[e2, :]
    return (x, y)

def tr_ps(image_points, im,imsave):  # impath
    size = im.shape
    global model_points
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, cmat, dpr,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)
    #print "Rotation Vector:\n {0}".format(rotation_vector)
    #print "Translation Vector:\n {0}".format(translation_vector)
    str1 = "Rotation Vector:\n {0}".format(rotation_vector)
    str2 = "Translation Vector:\n {0}".format(translation_vector)
    # Project a 3D point (0, 0, 100.0) onto the image plane.
    # We use this to draw a line sticking out of the nose
    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 100.0)]), rotation_vector, translation_vector,
                                                     cmat, dpr)
    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (255, 0, 0), -1)
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(im, p1, p2, (255, 0, 0), 2)
    # write info on  image
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (50, 50)
    org2 = (50, 100)
    # fontScale
    fontScale = 0.5
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 1
    # Using cv2.putText() method
    im = cv2.putText(im, str1, org, font, fontScale, color, thickness, cv2.LINE_AA)
    im = cv2.putText(im, str2, org2, font, fontScale, color, thickness, cv2.LINE_AA)
    # Display image
    #cv2.imshow("Output", im)
    cv2.imwrite(imsave,im)
    return success, rotation_vector, translation_vector
# reduce image acurecy
def accuracy_reduction(f1,d):
    if (d==1):
        f2 = cv2.pyrDown(f1)
        f3 = cv2.pyrDown(f2)
        f5 = cv2.pyrUp(f3)
        f6 = cv2.pyrUp(f5)
    else:
        f2 = cv2.pyrDown(f1)
        f6 = cv2.pyrUp(f2)
    return f6

# 3D model points.
model_points = np.array([
    (0.0, 0.0, 0.0),  # Nose tip
    (0.0, -48.0, 0),  # Chin
    (0.0, -48.0, 48.0),  # Left eye left corner
    (-48.0, -48.0, 48.0),  # Right eye right corne
    (-48.0, 0.0, 48.0),  # Left Mouth corner
    (-48.0, 0.0, 0.0)  # Right mouth corner

])
##
file_path = "F:" 
saveid = "F:"  
# the main
listOfImg = os.listdir(file_path)
for img in listOfImg:
    try:
        print (img)
        svpth = saveid + "\\" + "posture" + img
        frame = cv2.imread(file_path + "\\" + img)
        #frame = accuracy_reduction(frame,0)
        framec = calbration(frame)
        frameg = adjust_gamma(framec)
        frameh = histrans(frameg)
        framea1 = adpv_hsm(frameh)
        #
        filtred = clr_filt(framea1, framec)
        bfiltred = accuracy_reduction(filtred,0)
        features = gc2trk(bfiltred)
        ftr = np.array(features)
        sorted_corner = sortpoint(features)
        rot_tr_vec = tr_ps(sorted_corner,framec,svpth)
        
        cv2.destroyAllWindows()
        print ("  done")
    except:
        print("img "+img+ " error")

if cv2.waitKey(1) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

