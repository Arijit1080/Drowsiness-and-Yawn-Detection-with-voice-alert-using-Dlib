# Drowsiness and Yawn detection with voice alert using Dlib

Simple code in python to detect Drowsiness and Yawn and alert the user using Dlib.

## Dependencies

1. Python 3
2. opencv
3. dlib
4. imutils
5. scipy
6. numpy
7. argparse

## Run 

```
Python3 drowsiness_yawn.py -- webcam 0		//For external webcam, use the webcam number accordingly
```

## Setups

Change the threshold values according to your need
```
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 10`	//change this according to the distance from the camera
```

## Authors

**Arijit Das** 


## Acknowledgments

* https://www.pyimagesearch.com/



