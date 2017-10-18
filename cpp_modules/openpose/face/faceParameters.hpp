#ifndef OPENPOSE_FACE_FACE_PARAMETERS_HPP
#define OPENPOSE_FACE_FACE_PARAMETERS_HPP


namespace op
{
    const auto FACE_MAX_FACES = 96u;
    const auto FACE_NUMBER_PARTS = 68u;
    #define FACE_PAIRS_RENDER_GPU {0,1,  1,2,  2,3,  3,4,  4,5,  5,6,  6,7,  7,8,  8,9,  9,10,  10,11,  11,12,  12,13,  13,14,  14,15,  15,16,  17,18,  18,19,  19,20, \
                                  20,21,  22,23,  23,24,  24,25,  25,26,  27,28,  28,29,  29,30,  31,32,  32,33,  33,34,  34,35,  36,37,  37,38,  38,39,  39,40,  40,41, \
                                  41,36,  42,43,  43,44,  44,45,  45,46,  46,47,  47,42,  48,49,  49,50,  50,51,  51,52,  52,53,  53,54,  54,55,  55,56,  56,57,  57,58, \
                                  58,59,  59,48,  60,61,  61,62,  62,63,  63,64,  64,65,  65,66,  66,67,  67,60}
    const std::vector<unsigned int> FACE_PAIRS_RENDER {FACE_PAIRS_RENDER_GPU};
    #define FACE_COLORS_RENDER_GPU 255.f,    255.f,    255.f
    const std::vector<float> FACE_COLORS_RENDER{FACE_COLORS_RENDER_GPU};

    const std::vector<unsigned> EYE_INDEX = {
            36, 37, 38, 39, 40, 41,
            42, 43, 44, 45, 46, 47
    };

    const std::vector<unsigned> NOSE_INDEX = {
            27, 28, 29, 30, 31, 32, 33, 34, 35
    };

    const std::vector<unsigned> LIP_INDEX = {
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    };

    const std::vector<std::pair<unsigned, unsigned>> UPPER_LOWER_LIP_PAIRS = {
            {49, 59},
            {50, 58},
            {51, 57},
            {52, 56},
            {53, 55},
            {61, 67},
            {62, 66},
            {63, 65}
    };

    // Constant parameters
    const auto FACE_CCN_DECREASE_FACTOR = 8.f;
    const std::string FACE_PROTOTXT{"face/pose_deploy.prototxt"};
    const std::string FACE_TRAINED_MODEL{"face/pose_iter_116000.caffemodel"};

    // Rendering parameters
    const auto FACE_DEFAULT_ALPHA_KEYPOINT = 0.6f;
    const auto FACE_DEFAULT_ALPHA_HEAT_MAP = 0.7f;

}

#endif // OPENPOSE_FACE_FACE_PARAMETERS_HPP
