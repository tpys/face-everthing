#include <cstdio> // sscanf
#include <openpose/utilities/check.hpp>
#include <openpose/utilities/flagsToOpenPose.hpp>

namespace op
{

    ScaleMode flagsToScaleMode(const int keypointScale)
    {
        try
        {
            log("", Priority::Low, __LINE__, __FUNCTION__, __FILE__);
            if (keypointScale == 0)
                return ScaleMode::InputResolution;
            else if (keypointScale == 1)
                return ScaleMode::NetOutputResolution;
            else if (keypointScale == 2)
                return ScaleMode::OutputResolution;
            else if (keypointScale == 3)
                return ScaleMode::ZeroToOne;
            else if (keypointScale == 4)
                return ScaleMode::PlusMinusOne;
            // else
            const std::string message = "String does not correspond to any scale mode: (0, 1, 2, 3, 4) for (InputResolution,"
                                        " NetOutputResolution, OutputResolution, ZeroToOne, PlusMinusOne).";
            error(message, __LINE__, __FUNCTION__, __FILE__);
            return ScaleMode::InputResolution;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return ScaleMode::InputResolution;
        }
    }


    std::vector<HeatMapType> flagsToHeatMaps(const bool heatMapsAddParts, const bool heatMapsAddBkg, const bool heatMapsAddPAFs)
    {
        try
        {
            std::vector<HeatMapType> heatMapTypes;
            if (heatMapsAddParts)
                heatMapTypes.emplace_back(HeatMapType::Parts);
            if (heatMapsAddBkg)
                heatMapTypes.emplace_back(HeatMapType::Background);
            if (heatMapsAddPAFs)
                heatMapTypes.emplace_back(HeatMapType::PAFs);
            return heatMapTypes;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return {};
        }
    }

    RenderMode flagsToRenderMode(const int renderFlag, const int renderPoseFlag)
    {
        try
        {
            if (renderFlag == -1 && renderPoseFlag != -2)
                return flagsToRenderMode(renderPoseFlag, -2);
            else if (renderFlag == 0)
                return RenderMode::None;
            else if (renderFlag == 1)
                return RenderMode::Cpu;
            else if (renderFlag == 2)
                return RenderMode::Gpu;
            // else
            error("Undefined RenderMode selected.", __LINE__, __FUNCTION__, __FILE__);
            return RenderMode::None;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return RenderMode::None;
        }
    }

    Point<int> flagsToPoint(const std::string& pointString, const std::string& pointExample)
    {
        try
        {
            Point<int> point;
            const auto nRead = sscanf(pointString.c_str(), "%dx%d", &point.x, &point.y);
            checkE(nRead, 2, "Invalid resolution format: `" +  pointString + "`, it should be e.g. `" + pointExample
                   + "`.", __LINE__, __FUNCTION__, __FILE__);
            return point;
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return Point<int>{};
        }
    }
}
