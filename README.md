# Detect-unhealthy-sitting-pose
## Introduction
A tool for monitoring and reminding people when they sit in unhealthy pose to prevent spinal diseases and myopia, using Machine Learning, Computer Vision, Streamlit
## Steps
Data is collected by taking pictures with a webcam.
The collected images will be passed through Mediapipe Pose to extract the coordinates of 33 points on the body into a dataset.
Use this dataset to train the model to identify right or poor sitting posture. The selected model is RandomForest.
Finally, deploy on streamlit. 
## Output
