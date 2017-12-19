"""
Example script using only the Face detector of Openpose.
"""

import PyOpenPose as OP
import time
import cv2

import numpy as np
import os

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]


def ComputeBB(face, padding=0.4):
    minX = np.min(face[:, 0])
    minY = np.min(face[:, 1])

    maxX = np.max(face[:, 0])
    maxY = np.max(face[:, 1])

    width = maxX - minX
    height = maxY - minY

    padX = width * padding / 2
    padY = height * padding / 2

    minX -= padX
    minY -= padY

    width += 2 * padX
    height += 2 * padY

    score = np.mean(face[:, 2])
    return score, [int(minX), int(minY), int(width), int(height)]


def run():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    imgSize = list(frame.shape)
    outSize = imgSize[1::-1]

    print("Net output size: ", outSize)

    download_heatmaps = False
    with_hands = False
    with_face = True
    op = OP.OpenPose((656, 368), (240, 240), tuple(outSize), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                     download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)

    actual_fps = 0
    paused = False
    delay = {True: 0, False: 1}
    newFaceBB = initFaceBB = faceBB = [240, 120, 150, 150]

    print("Entering main Loop. Put your hand into the box to start tracking")
    while True:
        start_time = time.time()
        try:
            ret, frame = cap.read()
            rgb = frame[:, :outSize[0]]

        except Exception as e:
            print("Failed to grab", e)
            break

        t = time.time()
        op.detectFace(rgb, np.array(faceBB, dtype=np.int32).reshape((1, 4)))
        t = time.time() - t
        op_fps = 1.0 / t

        res = op.render(rgb)
        cv2.putText(res, 'UI FPS = %f, OP-FACE FPS = %f. Press \'r\' to reset.' % (actual_fps, op_fps), (20, 20), 0, 0.5,
                    (0, 0, 255))

        cv2.rectangle(res, (faceBB[0], faceBB[1]), (faceBB[0] + faceBB[2], faceBB[1] + faceBB[3]), [50, 155, 50], 2)
        cv2.rectangle(res, (newFaceBB[0], newFaceBB[1]), (newFaceBB[0] + newFaceBB[2], newFaceBB[1] + newFaceBB[3]),
                      [250, 55, 50], 1)
        cv2.imshow("OpenPose result", res)

        face = op.getKeypoints(op.KeypointType.FACE)[0].reshape(-1, 3)
        score, newFaceBB = ComputeBB(face)
        print("Res Score, faceBB: ", score, newFaceBB)
        if score > 0.5: # update BB only when score is good.
            faceBB = newFaceBB

        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused

        if key & 255 == ord('q'):
            break

        if key & 255 == ord('r'):
            faceBB = initFaceBB

        actual_fps = 1.0 / (time.time() - start_time)


if __name__ == '__main__':
    run()
