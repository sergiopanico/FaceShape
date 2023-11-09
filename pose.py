import cv2
import mediapipe as mp
import numpy as np


def normalize_image(img, normalized_length:int=640, square:bool=False):
    (h, w, _) = img.shape

    scale_factor = normalized_length/float(min(h,w))
    dim = (int(w*scale_factor), int(h*scale_factor))
    normalized_img = cv2.resize(img, dim ,cv2.INTER_AREA)
    if (square):
        normalized_img = normalized_img[int(abs(dim[1]-normalized_length)/2):normalized_length, int(abs(dim[0]-normalized_length)/2):normalized_length]

    return normalized_img

fontName = cv2.QT_FONT_NORMAL
fontSize = 0.4
fontColor = (0,0,255) #BGR
fontWeight = 1

#img = cv2.imread("D:\\git\\theproject\\media\\body\\spreadArms.jpg")
img = cv2.imread("D:\\git\\theproject\\media\\body\\WomanSpreadArms2.jpg")

base_options = mp.tasks.BaseOptions(model_asset_path='D:\\git\\theproject\\resources\\pose_landmarker_full.task')
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True,
    running_mode=mp.tasks.vision.RunningMode.IMAGE)
detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)

mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
pose_landmarker_result = detector.detect(mp_image)

imgH, imgW, _ = img.shape
normalizedLandmarks = [(int(l.x * imgW), int(l.y * imgH)) for l in pose_landmarker_result.pose_landmarks[0]]

(p11x, p11y) = normalizedLandmarks[11] #spalla sx
(p12x, p12y) = normalizedLandmarks[12] #spalla dx
(p15x, p15y) = normalizedLandmarks[15] #mano sinistra
(p16x, p16y) = normalizedLandmarks[16] #mano destra
(p23x, p23y) = normalizedLandmarks[23] #fianchi sinistra
(p24x, p24y) = normalizedLandmarks[24] #fianchi destra
leftmostX = min(normalizedLandmarks, key=lambda p:p[0])[0]
rightmostX = max(normalizedLandmarks, key=lambda p:p[0])[0]
waistMidpoint = (int)((p23y+p24y)/2)

offset = (int)(p11x-p12x)/2
if (p16x > p12x-offset or p15x < p11x+offset):
    raise Exception("Sembra che le braccia siano troppo attaccate al corpo")

imgMask = pose_landmarker_result.segmentation_masks[0].numpy_view()

def project_points_to_mask(m, p, v):
    (px,py) = p
    pj = px
    while m[py,px] >= 0.5 and pj > 0 and pj < m.shape[1]:
        px = px+(1*v)

    return (px,py) 

normalizedLandmarks.append(project_points_to_mask(imgMask,normalizedLandmarks[23],+1))
normalizedLandmarks.append(project_points_to_mask(imgMask,normalizedLandmarks[24],-1))


for lidx in range(0,len(normalizedLandmarks)):
    x = normalizedLandmarks[lidx][0]
    y = normalizedLandmarks[lidx][1]
    cv2.circle(img, (x, y), radius=2, color=(225, 0, 100), thickness=1)
    cv2.putText(img,f'{len(normalizedLandmarks)=}',(10, 30),fontFace=fontName,fontScale=fontSize*2,color=fontColor, thickness=fontWeight*2)
    cv2.putText(img,f'{lidx}',(x+2, y+2),fontFace=fontName,fontScale=fontSize,color=fontColor, thickness=fontWeight)


p12 = normalizedLandmarks[12]
p11 = normalizedLandmarks[11]
p34 = normalizedLandmarks[34]
p33 = normalizedLandmarks[33]

bodyLineCoeff = np.polyfit([p24x, p12[0]], [p24x, p12[1]], 1)
cv2.putText(img,f'{bodyLineCoeff[0]=}',(10, 60),fontFace=fontName,fontScale=fontSize*1.3,color=fontColor, thickness=fontWeight*2)

cv2.line(img,p12, p11, (0,255,0), 2)
cv2.line(img,p12, p34, (0,255,0), 2)
cv2.line(img,p11, p33, (0,255,0), 2)
cv2.line(img,p34, p33, (0,255,0), 2)

cv2.imshow("pose", img)
cv2.waitKey()
#cv2.imwrite("D:\\git\\theproject\\media\\body\\spreadArms_landmarks.JPG", img)
cv2.imwrite("D:\\git\\theproject\\media\\body\\WomanSpreadArms2_landmarks.JPG", img)

