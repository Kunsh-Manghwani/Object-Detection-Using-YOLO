import os
name = input("Enter the Name of Image:- ").strip()
com = 'python yolo_detect_image.py --image ' + name
os.system('cmd /k' + com)
