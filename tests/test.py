# Import packages
import os
import cv2
import sys
import imutils
from tensorflow.keras.models import load_model

# Import solver
path = os.getcwd()
path = path[:path.rfind('\\') + 1]
sys.path.insert(1, path + 'src')
from solver import solve_sudoku

# Load image
image_name = 'test0.jpg' # 'test1.jpg'
image = cv2.imread(image_name)
image = imutils.resize(image, width=600)

# Load recognition model
model = load_model(path + r'models\digit_classifier.h5')

# Solve sudoku
solution = solve_sudoku(image, model)

# Save it
# cv2.imwrite('result0.jpg', solution)  
# cv2.imwrite('result1.jpg', solution)  

# Show it
cv2.imshow('Solved puzzle', solution)
cv2.waitKey(0)