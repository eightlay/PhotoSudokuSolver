# Import packages
import cv2
import imutils
import numpy as np
from typing import Tuple
from skimage.segmentation import clear_border
from imutils.perspective import four_point_transform


def find_puzzle(image: np.ndarray) -> Tuple[np.ndarray]:
    '''
        Finds sudoku puzzle on the image and returns its perspective-transformed version (natural and grayscale)

        Parameters
        ----------
        - image : np.ndarray
                  image containing sudoku puzzle

        Returns
        -------
        Tuple[np.ndarray] {puzzle, grayscale puzzle}
    '''
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Blur image
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Invert colors
    thresh = cv2.bitwise_not(thresh)
    
    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # Sort by size in descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Sudoku contour
    puzzleCnt = None

    # Loop over the contours
    for c in cnts:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Assume that 4-points contour is the sudoku one
        if len(approx) == 4:
            puzzleCnt = approx
            break

    if puzzleCnt is None:
        raise Exception(("Sudoku is not found."))

    # Four point perspective transform
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))
    
    # Return results
    return (puzzle, warped)


def extract_digit(cell: np.ndarray) -> np.ndarray:
    '''
        Extract digit from cell

        Parameters
        ----------
        - cell - np.ndarray
                 image, which contains (or not) digit
        
        Returns
        -------
        np.ndarray {digit without any noise around}
    '''
    # Apply automatic thresholding
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # Clear borders
    thresh = clear_border(thresh)

    # Find contours
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # It is empty cell, if no contours found
    if len(cnts) == 0:
        return None

    # Mask largest contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # Percentage of masked pixels
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)

    # If percentage is less then 3% - it is noise we can ignore
    if percentFilled < 0.03:
        return None

    # Apply mask to thresh
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Return digit
    return digit
