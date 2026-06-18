# Eye Landmark Detection

## Overview

Eye Landmark Detection is a computer vision project that uses OpenCV and MediaPipe Face Mesh to detect facial landmarks and extract eye coordinates in real time from a webcam feed. The system identifies key landmark points around both eyes and visualizes them on the video stream while providing structured coordinate data for further analysis.

This project serves as a foundation for applications such as gaze estimation, blink detection, eye tracking, human-computer interaction, and driver monitoring systems.

---

## Objectives

* Detect facial landmarks in real time.
* Extract left and right eye landmark coordinates.
* Visualize eye landmarks on the webcam feed.
* Generate structured eye coordinate data in JSON format.
* Support moderate head tilts and users wearing glasses.

---

## Features

* Real-time webcam processing
* Facial landmark detection using MediaPipe Face Mesh
* Eye landmark extraction
* Landmark visualization using OpenCV
* JSON-based output format
* Lightweight and efficient implementation
* Beginner-friendly project structure

---

## Technology Stack

| Technology | Purpose                            |
| ---------- | ---------------------------------- |
| Python     | Core programming language          |
| OpenCV     | Image processing and webcam access |
| MediaPipe  | Facial landmark detection          |
| JSON       | Structured output representation   |

---

## Project Architecture

1. Capture video frames from the webcam.
2. Detect facial landmarks using MediaPipe Face Mesh.
3. Extract predefined landmarks corresponding to the left and right eyes.
4. Convert landmark positions into pixel coordinates.
5. Draw landmark points on the video frame.
6. Store and display eye coordinates in JSON format.

---

## Installation

### Prerequisites

* Python 3.8 or higher
* Webcam

### Install Dependencies

```bash
pip3 install opencv-python mediapipe
```

---

## Running the Project

Execute the following command:

```bash
python3 eye_landmark.py
```

Press **ESC** to exit the application.

---

## Output Format

Example JSON output:

```json
{
  "left_eye": [
    [320, 240],
    [325, 238],
    [330, 236]
  ],
  "right_eye": [
    [450, 240],
    [455, 238],
    [460, 236]
  ]
}
```

---

## Project Structure

```text
Eye_Landmark_Detection/
│
├── eye_landmark.py
├── README.md
├── requirements.txt
└── output.json
```

---

## Challenges Addressed

* Real-time facial landmark detection
* Eye localization under varying lighting conditions
* Detection for users wearing glasses
* Handling moderate head rotations and tilts
* Accurate coordinate extraction

---

## Applications

* Eye Tracking Systems
* Gaze Analysis
* Blink Detection
* Driver Monitoring Systems
* Human-Computer Interaction
* Accessibility Technologies

---

## Future Enhancements

* Gaze direction estimation
* Blink detection and counting
* Eye movement analytics
* Fatigue detection
* Multi-face eye tracking
* Integration with machine learning models

---

## Conclusion

This project demonstrates the practical application of computer vision techniques for extracting eye landmarks in real time. By leveraging OpenCV and MediaPipe, the system provides accurate eye coordinate detection and establishes a foundation for advanced eye-tracking and gaze-analysis applications.

---

## Author

**Piyush Kumar**
B.Tech (AI/ML) Student
Eye Landmark Detection Project
