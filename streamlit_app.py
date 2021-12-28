# import libraries
import numpy as np
import cv2
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
import matplotlib.pyplot as plt
import sklearn
import pickle
from pygame import mixer
import streamlit as st
import warnings

#Load model  
loaded_model = pickle.load(open('randomforest_model.pkl', 'rb'))
#import playsound 
short_tone = 'sound\short_tone.mp3'
long_tone = 'sound\long_tone.mp3'
# size image
train_img_height = 270
train_img_width = 360

st.markdown("<h1 style='text-align: center; color: #EC7063;'>SITTING POSE WATCHER</h1>", unsafe_allow_html=True)
st.image('pic\ergonomic-tips-students_change_size.png')
menu = ('Knowledge', 'Sitting pose watcher') 
st.sidebar.subheader('Choose an option to redirect')
choice = st.sidebar.radio('', menu)
if choice == 'Knowledge':
    st.header('What is a good sitting body position?')
    st.write('','''There is no one or single body position that is recommended for sitting. Every worker can sit comfortably by adjusting the angles of their hips, knees,
    ankles and elbows. The following are general recommendations. Occasional changes beyond given ranges are acceptable and sometimes beneficial.
    - Keep the joints such as hips, knees and ankles open slightly (more than 90°)
    - Keep knee joints at or below the hip joints
    - Keep ankle joints in front of the knees
    - Keep a gap the width of three fingers between the back of the knee joint and the front edge of the chair
    - Keep feet flat on the floor or on a foot rest
    - Keep the upper body within 30° of an upright position
    - Keep the lumbar support of the back rest in your lumbar region (around the waistband)
    - Always keep the head aligned with the spine
    - Keep upper arms between vertical and 20° forward
    - Keep elbows at an angle between 90° and 120°
    - Keep forearms between horizontal and 20° up
    - Place the working object so that it can be seen at viewing angle of 10° to 30° below the line of sight
    ''')
    st.image('pic\good_pose.png')
    st.markdown('**_But, it is important not to sit too long regardless of posture. A good idea is to get up and walk around a bit every 30 minutes._**')
    st.caption('source: https://www.ccohs.ca/oshanswers/ergonomics/sitting/sitting_position.html')
    
elif choice == 'Sitting pose watcher': 
    run = st.checkbox('Show Webcam')
    st.write('But, it is important not to sit too long regardless of posture. A good idea is to get up and walk around a bit')
    remind = st.checkbox('Remind me stand up')
    break_time = st.slider('Remind me to stand up every: (minutes)', 20, 60, 30, step = 5)
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(0)
    count = 0
    count_sitting = 0
    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.2) as pose:
        while run:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())    
        
        
            # Recognize right/wrong pose
            # Check the number of landmarks and take pose landmarks.
            pose_landmarks = results.pose_landmarks   
            if pose_landmarks is not None:
                assert len(pose_landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(len(pose_landmarks.landmark))
                pose_landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark]
                pose_landmarks *= np.array([train_img_width, train_img_height, train_img_width])
                pose_landmarks = np.around(pose_landmarks, 5).flatten().astype(str).tolist()
                y_pred = loaded_model.predict([pose_landmarks])
                y_pred_proba = loaded_model.predict_proba([pose_landmarks])
                if y_pred[0] == 'No sitting pose':
                    cv2.putText(image, str('Adjust the camera to see the entire human trunk'), (20,30), cv2.FONT_HERSHEY_SIMPLEX,  0.6, (0,0,255), 2, cv2.LINE_AA)
                elif y_pred[0] == 'Right pose':
                    cv2.putText(image, str(y_pred[0])+','+str(y_pred_proba[0].max()), (200,30), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,255,0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, str(y_pred[0])+','+str(y_pred_proba[0].max()), (200,30), cv2.FONT_HERSHEY_SIMPLEX,  1, (0,0,255), 2, cv2.LINE_AA)
            # Set warning sound if maintaining wrong pose more than 5s
                if y_pred[0] == 'Wrong pose':
                    count +=1
                    if count == 12*10: # 13 frames/1s
                        mixer.init() # initiate the mixer instance
                        mixer.music.load(short_tone) # loads the music, can be also mp3 file.
                        mixer.music.play() #play music
                        count = 0
                else:
                    count = 0
            # Set warning sound if maintaining sitting pose more than standard period
                if remind:    
                    if y_pred[0] != 'No sitting pose':
                        count_sitting +=1
                        if count_sitting == break_time*12*60: # 13 frames/1s
                            mixer.init() # initiate the mixer instance
                            mixer.music.load(long_tone) # loads the music, can be also mp3 file.
                            mixer.music.play() #play music
                            count_sitting = 0
                    else:
                        count_sitting = 0                            

            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(image)       

            if cv2.waitKey(1) & 0xFF == 27:
                break
        else:
            cap.release()
            cv2.destroyAllWindows()