//
// Created by padeler on 21/6/2017.
//

#include "OpenPoseWrapper.h"

#include <glog/logging.h>

#include <openpose/core/headers.hpp>
#include <openpose/core/scaleAndSizeExtractor.hpp>
#include <openpose/gui/headers.hpp>

#include <openpose/pose/headers.hpp>
#include <openpose/face/headers.hpp>
#include <openpose/hand/headers.hpp>

#include <openpose/utilities/headers.hpp>

#include <boost/throw_exception.hpp>

// Detector specific paramters are here:
//#include <openpose/hand/handParameters.hpp>
//#include <openpose/face/faceParameters.hpp>
//#include <openpose/pose/poseParameters.hpp>

struct OpenPoseWrapper::PrivateData
{
    PrivateData(const op::Point<int> &netInputSize, const op::Point<int> &netInputSizeFaceAndHands,
                const op::Point<int> &outputSize, const op::PoseModel &poseModel,
                const std::string &modelFolder, int numScales, float scaleGap, float blendAlpha,
                const std::vector<op::HeatMapType> &heatMapTypes, const op::ScaleMode &heatMapScale, int gpuId):
            poseExtractorCaffe{poseModel, modelFolder, gpuId, heatMapTypes, heatMapScale},
            poseRenderer{poseModel, nullptr, 0.05, true, blendAlpha},
            scaleAndSizeExtractor{netInputSize, outputSize, numScales, scaleGap},

            faceExtractor{netInputSizeFaceAndHands, netInputSizeFaceAndHands, modelFolder, gpuId, heatMapTypes, heatMapScale},
            faceRenderer{0.4},
            faceDetector(poseModel),

            handDetector(poseModel),
            handRenderer{0.2},
            handExtractor{netInputSizeFaceAndHands, netInputSizeFaceAndHands, modelFolder, gpuId, 1, 0.4, heatMapTypes, heatMapScale}

    {}

    op::CvMatToOpInput cvMatToOpInput;
    op::CvMatToOpOutput cvMatToOpOutput;
    op::PoseExtractorCaffe poseExtractorCaffe;

    op::PoseGpuRenderer poseRenderer;

    op::FaceExtractorCaffe faceExtractor;
    op::FaceDetector faceDetector;
    op::FaceGpuRenderer faceRenderer;

    op::HandExtractorCaffe handExtractor;
    op::HandDetector handDetector;
    op::HandGpuRenderer handRenderer;

    op::OpOutputToCvMat opOutputToCvMat;
    op::ScaleAndSizeExtractor scaleAndSizeExtractor;
};

OpenPoseWrapper::OpenPoseWrapper(const cv::Size &netPoseSize, const cv::Size &netFaceHandsSize, const cv::Size &outSize,
                                 const std::string &model, const std::string &modelFolder, const int logLevel,
                                 bool downloadHeatmaps, OpenPoseWrapper::ScaleMode scaleMode, bool withFace, bool withHands, int gpuId):withFace(withFace), withHands(withHands) {
//    google::InitGoogleLogging("OpenPose Wrapper");

    // Step 1 - Set logging level
    // - 0 will output all the logging messages
    // - 255 will output nothing

    op::check(0 <= logLevel && logLevel <= 255, "Wrong logging_level value.", __LINE__, __FUNCTION__, __FILE__);
    op::ConfigureLog::setPriorityThreshold((op::Priority)logLevel);

    // Step 2 - Init params
    op::Point<int> outputSize(outSize.width,outSize.height);
    op::Point<int> netInputSize(netPoseSize.width,netPoseSize.height);
    op::Point<int> netInputSizeFaceAndHands(netFaceHandsSize.width,netFaceHandsSize.height);

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
    membersPtr = std::shared_ptr<PrivateData>(new PrivateData(netInputSize, netInputSizeFaceAndHands,
                                                              outputSize, poseModel, modelFolder,
                                                              numScales, scaleGap, blendAlpha,
                                                              hmt, (op::ScaleMode)scaleMode, gpuId));

    // Step 4 - Initialize resources on desired thread (in this case single thread, i.e. we init resources here)
    membersPtr->poseExtractorCaffe.initializationOnThread();
    membersPtr->poseRenderer.initializationOnThread();
    if(withFace) {
        membersPtr->faceExtractor.initializationOnThread();
        membersPtr->faceRenderer.initializationOnThread();
    }
    if(withHands) {
        membersPtr->handExtractor.initializationOnThread();
        membersPtr->handRenderer.initializationOnThread();
    }
}

void OpenPoseWrapper::detectPose(const cv::Mat &rgb) {
    // Step 2 - Format input image to OpenPose input and output formats

    const op::Point<int> imageSize{rgb.cols, rgb.rows};
    std::vector<double> scaleInputToNetInputs;
    std::vector<op::Point<int>> netInputSizes;
    double scaleInputToOutput;
    op::Point<int> outputResolution;
    std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
            = membersPtr->scaleAndSizeExtractor.extract(imageSize);
    std::vector<op::Array<float>> netInputArray = membersPtr->cvMatToOpInput.createArray(rgb, scaleInputToNetInputs, netInputSizes);

    // Step 3 - Estimate poseKeypoints
    membersPtr->poseExtractorCaffe.forwardPass(netInputArray, imageSize, scaleInputToNetInputs);
}

void OpenPoseWrapper::detectFace(const cv::Mat &rgb) {
    if(!withFace)
    {
        BOOST_THROW_EXCEPTION(std::runtime_error("Face network was not initialized."));
    }
    const auto poseKeypoints = membersPtr->poseExtractorCaffe.getPoseKeypoints();
    const auto faceRectsOP = membersPtr->faceDetector.detectFaces(poseKeypoints, 1.0f);

    this->faceRects = cv::Mat(faceRectsOP.size(), 4, CV_32SC1, cv::Scalar(0));
    cv::Mat fr = this->faceRects.reshape(4,faceRects.rows); // stupid cv::Mat iterator cannot iterate over rows.
    std::transform(faceRectsOP.begin(), faceRectsOP.end(), fr.begin<cv::Vec4i>(), [](const op::Rectangle<float> &r) -> cv::Vec4i { return cv::Vec4i(r.x, r.y, r.width, r.height);});

    membersPtr->faceExtractor.forwardPass(faceRectsOP, rgb, 1.0f);
}

void OpenPoseWrapper::detectFace(const cv::Mat &rgb, const cv::Mat &faceRects)
{
    if(!withFace)
    {
        BOOST_THROW_EXCEPTION(std::runtime_error("Face network was not initialized."));
    }
    if(faceRects.cols!=4 or faceRects.type()!=CV_32SC1)
    {
        BOOST_THROW_EXCEPTION(std::runtime_error("Invalid face rectangles format. Expected Nx4 mat with type CV_32SC1"));
    }

    this->faceRects = faceRects; // keep a copy
    std::vector<op::Rectangle<float> > faceRectsOP(faceRects.rows);
    cv::Mat fr = faceRects.reshape(4,faceRects.rows); // stupid cv::Mat iterator cannot iterate over rows.
    std::transform(fr.begin<cv::Vec4i>(), fr.end<cv::Vec4i>(), faceRectsOP.begin(),
                   [](const cv::Vec4i &r) -> op::Rectangle<float> { return op::Rectangle<float>(r[0], r[1], r[2], r[3]);});

    membersPtr->faceExtractor.forwardPass(faceRectsOP, rgb, 1.0f);
}

void OpenPoseWrapper::detectHands(const cv::Mat &rgb) {
    if(!withHands)
    {
        BOOST_THROW_EXCEPTION(std::runtime_error("Hand network was not initialized."));
    }

    const auto poseKeypoints = membersPtr->poseExtractorCaffe.getPoseKeypoints();
    const auto handRectsOP = membersPtr->handDetector.detectHands(poseKeypoints, 1.0f);

    this->handRects= cv::Mat(handRectsOP.size(), 8, CV_32SC1, cv::Scalar(0));
    cv::Mat hr = this->handRects.reshape(8,handRects.rows); // stupid cv::Mat iterator cannot iterate over rows.
    std::transform(handRectsOP.begin(), handRectsOP.end(), hr.begin<cv::Vec8i>(),
                   [](const std::array<op::Rectangle<float>, 2> &r) -> cv::Vec8i
                   { return cv::Vec8i(r[0].x, r[0].y, r[0].width, r[0].height, r[1].x, r[1].y, r[1].width, r[1].height); });

    membersPtr->handExtractor.forwardPass(handRectsOP, rgb, 1.0f);
}

void OpenPoseWrapper::detectHands(const cv::Mat &rgb, const cv::Mat &handRects)
{
    if(!withHands)
    {
        BOOST_THROW_EXCEPTION(std::runtime_error("Hand network was not initialized."));
    }
    if(handRects.cols!=8 or handRects.type()!=CV_32SC1)
    {
        BOOST_THROW_EXCEPTION(std::runtime_error("Invalid hand rectangles format. Expected Nx8 mat with type CV_32SC1"));
    }

    this->handRects = handRects;
    std::vector<std::array<op::Rectangle<float>, 2> > handRectsOP(handRects.rows);
    cv::Mat hr = handRects.reshape(8,handRects.rows); // stupid cv::Mat iterator cannot iterate over rows.
    std::transform(hr.begin<cv::Vec8i>(), hr.end<cv::Vec8i>(), handRectsOP.begin(),
                   [](const cv::Vec8i &r) -> std::array<op::Rectangle<float>, 2>
                   { return std::array<op::Rectangle<float>, 2>{op::Rectangle<float>(r[0], r[1], r[2], r[3]), op::Rectangle<float>(r[4], r[5], r[6], r[7])};});

    membersPtr->handExtractor.forwardPass(handRectsOP, rgb, 1.0f);
}

cv::Mat OpenPoseWrapper::render(const cv::Mat &rgb)
{
    op::Array<float> outputArray;

    const op::Point<int> imageSize{rgb.cols, rgb.rows};
    std::vector<double> scaleInputToNetInputs;
    std::vector<op::Point<int>> netInputSizes;
    double scaleInputToOutput;
    op::Point<int> outputResolution;
    std::tie(scaleInputToNetInputs, netInputSizes, scaleInputToOutput, outputResolution)
            = membersPtr->scaleAndSizeExtractor.extract(imageSize);

    outputArray = membersPtr->cvMatToOpOutput.createArray(rgb, scaleInputToOutput, outputResolution);

    const auto poseKeypoints = membersPtr->poseExtractorCaffe.getPoseKeypoints();
    membersPtr->poseRenderer.renderPose(outputArray, poseKeypoints, scaleInputToOutput);

    if(withFace){
        const auto faceKeypoints = membersPtr->faceExtractor.getFaceKeypoints();
        membersPtr->faceRenderer.renderFace(outputArray, faceKeypoints);
    }
    if(withHands) {
        const auto handKeypoints = membersPtr->handExtractor.getHandKeypoints();
        membersPtr->handRenderer.renderHand(outputArray, handKeypoints);
    }

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


std::vector<cv::Mat> OpenPoseWrapper::getHandHeatmaps() {

    std::array<op::Array<float>, 2> maps = membersPtr->handExtractor.getHeatMaps();

    std::vector<cv::Mat> res;
    res.push_back(maps.at(0).getConstCvMat().clone());
    res.push_back(maps.at(1).getConstCvMat().clone());
    return res;
}

cv::Mat OpenPoseWrapper::getFaceHeatmaps() {
    op::Array<float> maps = membersPtr->faceExtractor.getHeatMaps();
    return maps.getConstCvMat().clone();
}
