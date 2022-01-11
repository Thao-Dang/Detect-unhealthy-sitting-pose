# Detect-unhealthy-sitting-pose
## Introduction
A tool for monitoring and reminding people when they sit in unhealthy pose to prevent spinal diseases and myopia, using Machine Learning, Computer Vision, Streamlit
## Steps
Data is collected by taking pictures with a webcam.
The collected images will be passed through Mediapipe Pose to extract the coordinates of 33 points on the body into a dataset.
Use this dataset to train the model to identify right or poor sitting posture. The selected model is RandomForest.
Finally, deploy on streamlit. 
## Output
https://user-images.githubusercontent.com/90619603/148952764-836da1f9-988e-462c-86f4-19931c295198.mp4

