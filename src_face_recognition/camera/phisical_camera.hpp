#pragma once
#include "godot_cpp/godot.hpp"
#include "godot_cpp/classes/node.hpp"
#include "godot_cpp/classes/image_texture.hpp"
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
#include <iostream>


namespace EyeLights { 
namespace EyeRecognizer {
    
class PhisicalCamera : public godot::Node  {
    public:
        PhisicalCamera();
        ~PhisicalCamera();
        void _init();
        void _process(double delta) override;
        void shutdown();
        void create_imagetexture_objects();
        double delta(cv::Mat &img, int i, int j, int k, int l);
        int vectorValue(cv::Mat &img, int i, int j);
        bool open(int cameraId);
        cv::Mat get_current_frame();
        static void _bind_methods();

    private: // Godot interfaces.
        GDCLASS(PhisicalCamera, godot::Node);
        cv::Mat currentFrame;
        cv::VideoCapture cap;
        int cameraID {-1};
        bool isOpened {false};
        godot::Ref<godot::ImageTexture> texture;
        godot::Ref<godot::Image> image;
    };
}}
