"""
Example script using only the Hand detector of Openpose.
"""

import PyOpenPose as OP
import time
import cv2

import numpy as np
import os
from matplotlib import pyplot as plt

from OpLoop_heatmaps_example import showHeatmaps

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]


def ComputeBB(hand, padding=1.5):
    minX = np.min(hand[:, 0])
    minY = np.min(hand[:, 1])

    maxX = np.max(hand[:, 0])
    maxY = np.max(hand[:, 1])

    width = maxX - minX
    height = maxY - minY

    cx = minX + width/2
    cy = minY + height/2

    width = height = max(width, height)
    width = height = width * padding

    minX = cx - width/2
    minY = cy - height/2


    score = np.mean(hand[:, 2])
    return score, [int(minX), int(minY), int(width), int(height)]


def run():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    imgSize = list(frame.shape)
    outSize = imgSize[1::-1]

    print("Net output size: ", outSize)

    download_heatmaps = True
    with_hands = True
    with_face = False
    op = OP.OpenPose((656, 368), (240, 240), tuple(outSize), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                     download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)

    actual_fps = 0
    paused = False
    delay = {True: 0, False: 1}
    newHandBB = initHandBB = handBB = [270, 190, 200, 200]

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
        op.detectHands(rgb, np.array(handBB + [0, 0, 0, 0], dtype=np.int32).reshape((1, 8)))
        t = time.time() - t
        op_fps = 1.0 / t

        res = op.render(rgb)
        cv2.putText(res, 'UI FPS = %f, OP-HAND FPS = %f. Press \'r\' to reset.' % (actual_fps, op_fps), (20, 20), 0, 0.5,
                    (0, 0, 255))

        cv2.rectangle(res, (handBB[0], handBB[1]), (handBB[0] + handBB[2], handBB[1] + handBB[3]), [50, 155, 50], 2)
        cv2.rectangle(res, (newHandBB[0], newHandBB[1]), (newHandBB[0] + newHandBB[2], newHandBB[1] + newHandBB[3]),
                      [250, 55, 50], 1)

        if download_heatmaps:
            left_hands, right_hands = op.getHandHeatmaps()
            for pidx in range(len(left_hands)):
                hm = showHeatmaps(left_hands[pidx], "Left Hand"+str(pidx))
                plt.imshow(hm)

                hm = hm[:, ::-1]
                x,y,w,h = handBB
                hs = cv2.resize(hm,(w,h))
                hs3 = (np.dstack((hs,hs,hs)) * 255).astype(np.ubyte)
                res[y:y+h, x:x+w] += hs3


        cv2.imshow("OpenPose result", res)


        leftHand = op.getKeypoints(op.KeypointType.HAND)[0].reshape(-1, 3)
        score, newHandBB = ComputeBB(leftHand)
        print("Res Score, HandBB: ", score, newHandBB)
        # if score > 0.5: # update BB only when score is good.
        #     handBB = newHandBB

        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused

        if key & 255 == ord('q'):
            break

        if key & 255 == ord('r'):
            handBB = initHandBB
        if key & 255 == ord('u'):
            plt.show()

        actual_fps = 1.0 / (time.time() - start_time)


if __name__ == '__main__':
    run()
