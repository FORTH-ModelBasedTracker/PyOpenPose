"""
Example script using PyOpenPose.
"""
import PyOpenPose as OP
import time
import cv2

import numpy as np
import os

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]


def showPAFs(PAFs, startIdx=0, endIdx=16):

    for idx in range(startIdx, endIdx):
        X = PAFs[idx*2]
        Y = PAFs[idx*2+1]
        tmp = np.dstack((X, Y, np.zeros_like(X)))

        # tmp[X == 0] = 0
        # print "tmp: ", np.min(tmp), np.max(tmp)
        # bg = (hm[1] * 255).astype(np.ubyte)
        # tmp = cv2.normalize(tmp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
        cv2.imshow("PAF "+str(idx), tmp)


def run():

    cap = cv2.VideoCapture(0)

    # op = OP.OpenPose((656, 368), (368, 368), (1280, 720), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, False)
    op = OP.OpenPose((320, 240), (240, 240), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, True)

    actualFPS = 0
    paused = False
    delay = {True: 0, False: 1}

    frame = 0
    print "Entering main Loop."
    while True:
        loopStart = time.time() * 1000
        try:
            ret, frame = cap.read()
            rgb = frame

        except Exception as e:
            print "Failed to grab", e
            break

        t = time.time()
        op.detectPose(rgb)
        op.detectFace(rgb)
        op.detectHands(rgb)
        t = time.time() - t
        fps = 1.0 / t

        res = op.render(rgb)

        cv2.putText(res, 'UI FPS = %f, OP FPS = %f' % (actualFPS, fps), (20, 20), 0, 0.5, (0, 0, 255))

        persons = op.getKeypoints(op.KeypointType.POSE)[0]
        hm = op.getHeatmaps()
        parts = hm[:18]
        background = hm[18]
        PAFs = hm[19:] # each PAF has two channels (total 16 PAFs)
        cv2.imshow("Right Wrist", parts[4])
        cv2.imshow("background", background)

        showPAFs(PAFs)

        if persons is not None and len(persons) > 0:
            print "First Person: ", persons[0].shape

        cv2.imshow("OpenPose result", res)

        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused

        if key & 255 == ord('q'):
            break

        frame += 1
        loopEnd = time.time() * 1000
        actualFPS = (1000.0 / (loopEnd - loopStart))

if __name__ == '__main__':
    run()
