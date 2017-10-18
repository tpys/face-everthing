//
// Created by root on 17-9-25.
//

#include "LBF.h"


Params::Params(): bagging_overlap(0.4), max_numtrees(10), max_depth(5), landmark_num(68), initial_num(5), max_numstage(7),
             max_radio_radius{0.4,0.3,0.2,0.15, 0.12, 0.10, 0.08, 0.06, 0.06,0.05},
             max_numfeats{500, 500, 500, 300, 300, 200, 200,200,100,100}, max_numthreshs(500) {

}