"""
Example script using PyOpenPose.
"""
import PyMBVCore
import PyMBVAcquisition as acq
import PyOpenPose as OP
import time
import cv2

import numpy as np
import os

OPENPOSE_ROOT = os.environ["OPENPOSE_ROOT"]


def showHeatmaps(hm):
    for idx, h in enumerate(hm):
        cv2.imshow("HeatMap "+str(idx), h)


def showPAFs(PAFs, startIdx=0, endIdx=16):
    allpafs = []
    for idx in range(startIdx, endIdx):
        X = PAFs[idx*2]
        Y = PAFs[idx*2+1]
        tmp = np.dstack((X, Y, np.zeros_like(X)))
        allpafs.append(tmp)

    pafs = np.mean(allpafs, axis=0)
    cv2.imshow("PAF", pafs)


def run():

    cap = acq.OpenNIGrabber(True, True, '/home/mad/Development/Projects/BlenderMBV/Scripts/data/media/openni.xml', "")
    cap.initialize()
    download_heatmaps = True
    with_face = with_hands = False
    op = OP.OpenPose((320, 240), (368, 368), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0,
                     download_heatmaps, OP.OpenPose.ScaleMode.ZeroToOne, with_face, with_hands)
    # op = OP.OpenPose((320, 240), (240, 240), (640, 480), "COCO", OPENPOSE_ROOT + os.sep + "models" + os.sep, 0, download_heatmaps)


    actual_fps = 0
    paused = False
    delay = {True: 0, False: 1}

    print "Entering main Loop."
    while True:
        start_time = time.time()
        try:

            frames, calibs = cap.grab()
            rgb = frames[1]


        except Exception as e:
            print "Failed to grab", e
            break

        t = time.time()
        op.detectPose(rgb)
        # op.detectFace(rgb)
        # op.detectHands(rgb)
        t = time.time() - t
        op_fps = 1.0 / t

        res = op.render(rgb)
        cv2.putText(res, 'UI FPS = %f, OP FPS = %f' % (actual_fps, op_fps), (20, 20), 0, 0.5, (0, 0, 255))
        persons = op.getKeypoints(op.KeypointType.POSE)[0]

        if download_heatmaps:
            hm = op.getHeatmaps()
            print "HM ",hm.shape, hm.dtype
            showHeatmaps(hm)

            # hm = op.getHeatmaps()
            # parts = hm[:18]
            # background = hm[18]
            # PAFs = hm[19:]  # each PAF has two channels (total 16 PAFs)
            # cv2.imshow("Right Wrist", parts[4])
            # cv2.imshow("background", background)
            # showPAFs(PAFs)

        if persons is not None and len(persons) > 0:
            print "First Person: ", persons[0].shape

        cv2.imshow("OpenPose result", res)

        key = cv2.waitKey(delay[paused])
        if key & 255 == ord('p'):
            paused = not paused

        if key & 255 == ord('q'):
            break

        actual_fps = 1.0 / (time.time() - start_time)

if __name__ == '__main__':
    run()
