#ifndef TRTX_YOLOV5_UTILS_H_
#define TRTX_YOLOV5_UTILS_H_

#include <dirent.h>
#include <opencv2/opencv.hpp>
#include "yololayer.h"


static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));

    return out;
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}


void draw_mask(cv::Mat& cvt_img, cv::Mat& seg_res, cv::Mat& lane_res, std::vector<Yolo::Detection>& res)
{
    static const std::vector<cv::Vec3b> segColor{cv::Vec3b(0, 0, 0), cv::Vec3b(0, 255, 0), cv::Vec3b(255, 0, 0)};
    static const std::vector<cv::Vec3b> laneColor{cv::Vec3b(0, 0, 0), cv::Vec3b(0, 0, 255), cv::Vec3b(0, 0, 0)};
    // cv::Mat cvt_img_cpu;
    // cvt_img.download(cvt_img_cpu);

    // handling seg and lane results
    for (int row = 0; row < cvt_img.rows; ++row) {
        uchar* pdata = cvt_img.data + row * cvt_img.step;
        for (int col = 0; col < cvt_img.cols; ++col) {
            int seg_idx = seg_res.at<int>(row, col);
            int lane_idx = lane_res.at<int>(row, col);
            //std::cout << "enter" << ix << std::endl;
            for (int i = 0; i < 3; ++i) {
                if (lane_idx) {
                    if (i != 2)
                        pdata[i] = pdata[i] / 2 + laneColor[lane_idx][i] / 2;
                }
                else if (seg_idx)
                    pdata[i] = pdata[i] / 2 + segColor[seg_idx][i] / 2;
            }
            pdata += 3;
        }
    }
}

#endif  // TRTX_YOLOV5_UTILS_H_