# Import packages
import cv2
import numpy as np
from sudoku import Sudoku
from sudoku_reader import *
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array


def solve_sudoku(image: np.ndarray, recognition_model: Sequential)\
        -> np.ndarray:
    '''
        Solve sudoku and return solution drawn on origin prespective-transformed image

        Parameters
        ----------
        - image : np.ndarray
                  image with sudoku puzzle

        Returns 
        -------
        np.ndarray {solution of sudoku puzzle} 
    '''
    # Find puzzle
    (puzzle, warped) = find_puzzle(image)

    # Sudoku board
    board = np.zeros((9, 9), dtype="int")

    # Cell size
    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9

    # Store cell's location
    cellLocs = []

    # Loop over the board
    for y in range(0, 9):
        # Store cell's location to row
        row = []

        for x in range(0, 9):
            # Coords of cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            # Add cell to row
            row.append((startX, startY, endX, endY))

            # Extract digit from warped image
            cell = warped[startY:endY, startX:endX]
            digit = extract_digit(cell)

            # verify that the digit is not empty
            if digit is not None:
                # Prepare for recognition
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                # Recognize digit and store prediction
                pred = recognition_model.predict(roi).argmax(axis=1)[0]
                board[y, x] = pred

        # Store row
        cellLocs.append(row)

    # Solve sudoku
    solution = Sudoku(3, 3, board=board.tolist())
    solution = solution.solve()

    # Draw solution
    for row, rowval in enumerate(zip(cellLocs, solution.board)):
        (cellRow, boardRow) = rowval
        # loop over individual cell in the row
        for col, colval in enumerate(zip(cellRow, boardRow)):
            (box, digit) = colval

            # Draw digit if it is not on the origin image
            if board[row, col] == 0:
                # Cell coordinates
                startX, startY, endX, endY = box

                # compute the coordinates of where the digit will be drawn
                # on the output puzzle image
                textX = int((endX - startX) * 0.33)
                textY = int((endY - startY) * -0.2)
                textX += startX
                textY += endY

                # draw the result digit on the Sudoku puzzle image
                cv2.putText(puzzle, str(digit), (textX, textY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Return result
    return puzzle
