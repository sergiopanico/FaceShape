import cv2
import mediapipe as mp


def normalize_image(img, normalized_length:int=640, square:bool=False):
    (h, w, _) = img.shape

    scale_factor = normalized_length/float(min(h,w))
    dim = (int(w*scale_factor), int(h*scale_factor))
    normalized_img = cv2.resize(img, dim ,cv2.INTER_AREA)
    if (square):
        normalized_img = normalized_img[int(abs(dim[1]-normalized_length)/2):normalized_length, int(abs(dim[0]-normalized_length)/2):normalized_length]

    return normalized_img

mp_drawing = mp.solutions.drawing_utils
# Load the MediaPipe Face Mesh model
mp_face_mesh = mp.solutions.face_mesh

# Initialize MediaPipe Face Landmarker
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)


# Load the input image
image = cv2.imread('D:\\git\\theproject\\media\\selfies\\oval_01.jpg')

# Convert the image to RGB format
#image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process image and get face landmarks
results = face_mesh.process(image)

# Draw face landmarks on image
# annotated_image = image.copy()
# for face_landmarks in results.multi_face_landmarks:
#     mp_drawing.draw_landmarks(
#         image=annotated_image,
#         landmark_list=face_landmarks,
#         #connections=mp_face_mesh.FACE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
#         connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1))

# # Display annotated image
# cv2.imshow('MediaPipe Face Landmarks', annotated_image)

fontName = cv2.QT_FONT_NORMAL
fontSize = 0.4
fontColor = (0,0,255) #BGR
fontWeight = 1
for face in results.multi_face_landmarks:
    #lidx = 0
   # for landmark in face.landmark:
    for lidx in range(0,len(face.landmark)):
   # for lidx in [33,133,158,153,144,160]:
        landmark = face.landmark[lidx]
        x = landmark.x
        y = landmark.y

        shape = image.shape 
        relative_x = int(x * shape[1])
        relative_y = int(y * shape[0])

        cv2.circle(image, (relative_x, relative_y), radius=2, color=(225, 0, 100), thickness=1)
        cv2.putText(image,f'{len(face.landmark)=}',(10, 30),fontFace=fontName,fontScale=fontSize*2,color=fontColor, thickness=fontWeight*2)
        cv2.putText(image,f'{lidx}',(relative_x+2, relative_y+2),fontFace=fontName,fontScale=fontSize,color=fontColor, thickness=fontWeight)
        lidx = lidx+1

cv2.imshow('Landmark points', image)

cv2.imwrite("D:\\git\\theproject\\media\\selfies\\oval_01_landmarks.jpg", image)

cv2.waitKey(0)
# Release resources
face_mesh.close()





