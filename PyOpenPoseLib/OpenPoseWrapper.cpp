//
// Created by padeler on 21/6/2017.
//

#include "OpenPoseWrapper.h"

#include <glog/logging.h>

#include <openpose/core/headers.hpp>
#include <openpose/gui/headers.hpp>

#include <openpose/pose/headers.hpp>
#include <openpose/face/headers.hpp>
#include <openpose/hand/headers.hpp>

#include <openpose/utilities/headers.hpp>

// Detector specific paramters are here:
//#include <openpose/hand/handParameters.hpp>
//#include <openpose/face/faceParameters.hpp>
//#include <openpose/pose/poseParameters.hpp>

struct OpenPoseWrapper::PrivateData
{
    PrivateData(const op::Point<int> &netInputSize, const op::Point<int> &netOutputSize,
                const op::Point<int> &netInputSizeFace, const op::Point<int> &netOutputSizeFace,
                const op::Point<int> &outputSize, const op::PoseModel &poseModel,
                const std::string &modelFolder, int numScales, float scaleGap, float blendAlpha,
                const std::vector<op::HeatMapType> &heatMapTypes, const op::ScaleMode &heatMapScale):
            cvMatToOpInput{netInputSize, numScales, scaleGap}, cvMatToOpOutput{outputSize},
            opOutputToCvMat{outputSize},

            poseExtractorCaffe{netInputSize, netOutputSize, outputSize, numScales, poseModel, modelFolder, 0, heatMapTypes, heatMapScale},
            poseRenderer{netOutputSize, outputSize, poseModel, nullptr, 0.05, blendAlpha},

            faceExtractor{netInputSizeFace, netOutputSizeFace, modelFolder, 0},
            faceRenderer{outputSize, 0.4},
            faceDetector(poseModel),

            handDetector(poseModel),
            handRenderer{outputSize, 0.2},
            handExtractor{netInputSizeFace, netOutputSizeFace, modelFolder, 0}

    {}

    op::CvMatToOpInput cvMatToOpInput;
    op::CvMatToOpOutput cvMatToOpOutput;
    op::PoseExtractorCaffe poseExtractorCaffe;

    op::PoseRenderer poseRenderer;

    op::FaceExtractor faceExtractor;
    op::FaceDetector faceDetector;
    op::FaceRenderer faceRenderer;

    op::HandExtractor handExtractor;
    op::HandDetector handDetector;
    op::HandRenderer handRenderer;

    op::OpOutputToCvMat opOutputToCvMat;
};

OpenPoseWrapper::OpenPoseWrapper(const cv::Size &netPoseSize, const cv::Size &netFaceSize, const cv::Size &outSize, const std::string &model, const std::string &modelFolder, const int logLevel,
                                 bool downloadHeatmaps, OpenPoseWrapper::ScaleMode scaleMode) {
    google::InitGoogleLogging("OpenPose Wrapper");

    // Step 1 - Set logging level
    // - 0 will output all the logging messages
    // - 255 will output nothing

    op::check(0 <= logLevel && logLevel <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)logLevel);

    // Step 2 - Init params
    op::Point<int> outputSize(outSize.width,outSize.height);
    op::Point<int> netInputSize(netPoseSize.width,netPoseSize.height);
    op::Point<int> netOutputSize = netInputSize;
    op::Point<int> netInputSizeFace(netFaceSize.width,netFaceSize.height);
    op::Point<int> netOutputSizeFace = netInputSizeFace;

    op::PoseModel poseModel;

    op::log("", op::Priority::Low, __LINE__, __FUNCTION__, __FILE__);
    if (model == "COCO")
        poseModel = op::PoseModel::COCO_18;
    else if (model == "MPI")
        poseModel = op::PoseModel::MPI_15;
    else if (model == "MPI_4_layers")
        poseModel = op::PoseModel::MPI_15_4;
    else
    {
        op::error("String does not correspond to any model (COCO, MPI, MPI_4_layers)", __LINE__, __FUNCTION__, __FILE__);
        poseModel = op::PoseModel::COCO_18;
    }

    int numScales = 1;
    float scaleGap = 0.3; // not used if numScales==1
    float blendAlpha = 0.6;

    //if you need to download bodypart heatmaps, Background or PAFs. They must be enabled here.
    std::vector<op::HeatMapType> hmt = {};
    if(downloadHeatmaps)
    {
        hmt = {op::HeatMapType::Parts, op::HeatMapType::Background, op::HeatMapType::PAFs};
    }

    // Step 3 - Initialize all required classes
    membersPtr = std::shared_ptr<PrivateData>(new PrivateData(netInputSize, netOutputSize,
                                                              netInputSizeFace, netOutputSizeFace,
                                                              outputSize, poseModel, modelFolder,
                                                              numScales, scaleGap, blendAlpha,
                                                              hmt, (op::ScaleMode)scaleMode));

    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    membersPtr->poseExtractorCaffe.initializationOnThread();
    membersPtr->poseRenderer.initializationOnThread();

    membersPtr->faceExtractor.initializationOnThread();
    membersPtr->faceRenderer.initializationOnThread();

    membersPtr->handExtractor.initializationOnThread();
    membersPtr->handRenderer.initializationOnThread();
}

void OpenPoseWrapper::detectPose(const cv::Mat &rgb) {
    // Step 2 - Format input image to OpenPose input and output formats
    op::Array<float> netInputArray;
    std::vector<float> scaleRatios;
    std::tie(netInputArray, scaleRatios) = membersPtr->cvMatToOpInput.format(rgb);
    // Step 3 - Estimate poseKeypoints
    membersPtr->poseExtractorCaffe.forwardPass(netInputArray, {rgb.cols, rgb.rows}, scaleRatios);
}

void OpenPoseWrapper::detectFace(const cv::Mat &rgb) {
    const auto poseKeypoints = membersPtr->poseExtractorCaffe.getPoseKeypoints();
    const auto faceRects = membersPtr->faceDetector.detectFaces(poseKeypoints, 1.0f);
    membersPtr->faceExtractor.forwardPass(faceRects, rgb, 1.0f);
}

void OpenPoseWrapper::detectHands(const cv::Mat &rgb) {
    const auto poseKeypoints = membersPtr->poseExtractorCaffe.getPoseKeypoints();
    const auto handRects = membersPtr->handDetector.detectHands(poseKeypoints, 1.0f);
    membersPtr->handExtractor.forwardPass(handRects, rgb, 1.0f);
}


cv::Mat OpenPoseWrapper::render(const cv::Mat &rgb)
{
    double scaleInputToOutput;
    op::Array<float> outputArray;
    std::tie(scaleInputToOutput, outputArray) = membersPtr->cvMatToOpOutput.format(rgb);

    const auto poseKeypoints = membersPtr->poseExtractorCaffe.getPoseKeypoints();
    const auto faceKeypoints = membersPtr->faceExtractor.getFaceKeypoints();
    const auto handKeypoints = membersPtr->handExtractor.getHandKeypoints();

    // Step 4 - Render poseKeypoints
    membersPtr->poseRenderer.renderPose(outputArray, poseKeypoints);
    membersPtr->faceRenderer.renderFace(outputArray, faceKeypoints);
    membersPtr->handRenderer.renderHand(outputArray, handKeypoints);

    // Step 5 - OpenPose output format to cv::Mat
    auto outputImage = membersPtr->opOutputToCvMat.formatToCvMat(outputArray);
    return outputImage;
}

OpenPoseWrapper::KeypointGroups OpenPoseWrapper::getKeypoints(KeypointType kpt) {

    op::Array<float> faces,persons;
    std::array<op::Array<float>, 2> hands;

    KeypointGroups res;
    switch(kpt){
        case FACE:
            faces = membersPtr->faceExtractor.getFaceKeypoints();
            res.push_back(faces.getConstCvMat()); // all faces in a cv::Mat at index 0
            break;
        case HAND:
            hands = membersPtr->handExtractor.getHandKeypoints();
            res.push_back(hands[0].getConstCvMat()); // left hands cv::Mat
            res.push_back(hands[1].getConstCvMat()); // right hands cv::Mat
            break;
        default: // POSE
            persons = membersPtr->poseExtractorCaffe.getPoseKeypoints();
            res.push_back(persons.getConstCvMat()); // all persons in a cv::Mat at index 0
            break;
    }

    return res;
}

cv::Mat OpenPoseWrapper::getHeatmaps() {
    op::Array<float> maps = membersPtr->poseExtractorCaffe.getHeatMaps();
    return maps.getConstCvMat().clone();
}

