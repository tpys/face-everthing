#ifndef OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP
#define OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP

#include <openpose/core/common.hpp>
#include <openpose/core/enumClasses.hpp>

namespace op
{

    OP_API ScaleMode flagsToScaleMode(const int keypointScale);

    OP_API std::vector<HeatMapType> flagsToHeatMaps(const bool heatMapsAddParts = false, const bool heatMapsAddBkg = false,
                                                    const bool heatMapsAddPAFs = false);

    OP_API RenderMode flagsToRenderMode(const int renderFlag, const int renderPoseFlag = -2);

    OP_API Point<int> flagsToPoint(const std::string& pointString, const std::string& pointExample = "1280x720");
}

#endif // OPENPOSE_UTILITIES_FLAGS_TO_OPEN_POSE_HPP
