#include "phisical_camera.hpp"

namespace EyeLights { namespace EyeRecognizer {
    using namespace godot;
    using namespace std;
    using namespace cv;

    PhisicalCamera::PhisicalCamera() {
        image.instantiate();
        texture.instantiate();
    }

    PhisicalCamera::~PhisicalCamera() {

    }

    void PhisicalCamera::_bind_methods() {
        std::cout << "PhisicalCamera::_bind_methods()" << std::endl;
        ClassDB::bind_method(D_METHOD("open"), &PhisicalCamera::open);
    }

    bool PhisicalCamera::open(int cameraId) {

        // Load the trained model
        cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
        model->read("/home/eyelights/Documents/face_recognition/model.xml");

        // Define the labels and corresponding person names
        std::vector<int> labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<std::vector<cv::String>> user = {{"Frederic_Aubert","Sport"}, {"Jean-Pierre_Stang","Comfort"}, {"Jules_Gregoire","Eco"},
            {"Neyl_Boukerche","Maximum Range"}, {"Pascal_Chevallier","Sport"}, {"Alexandre_Goudeau","Comfort"}, {"Jerome_Pauc","Eco"},
            {"Ilia_Seliverstov","Maximum Range"}, {"Stephane_Grange","Sport"}, {"Yannick_Durindel","Comfort"}, {"Nicolas_Hourcastagnou","Eco"} };

        // Open the camera
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open the camera." << std::endl;
            return 1;
        }

        cv::Mat frame;
        cv::Mat image;
        cv::Mat faceROI;
        cv::String mode;

        while (true) {
            // Read a frame from the camera
            cap.read(frame);
            resize(frame, image, Size(500, 500));
            cvtColor(image, faceROI, cv::COLOR_BGR2GRAY); // Convert to grayscale
            
            int predictedLabel = -1;
            double Confidence = 0.0;
            model->predict(faceROI, predictedLabel, Confidence);

            // Display the predicted person name and confidence
            if (predictedLabel!=-1) {
                std::cout << "Name: " << user[predictedLabel][0] << std::endl;
                if (mode!=user[predictedLabel][1]) {
                    cv::String mode = user[predictedLabel][1];
                    std::cout << "Switching to mode " << mode << std::endl;
                }
                std::cout << "Confidence: " << std::to_string(int(Confidence*100)/100) << "%" << std::endl;
            }
            //std::cout << mode << std::endl;
        }
        return 0;
    }


    void PhisicalCamera::_process(double delta) {
        if (isOpened) {
            cap >> currentFrame;
            if (!currentFrame.empty()) {
                cv::imshow("Camera", currentFrame);
                cv::waitKey(1);
            }
        }
    }

    void PhisicalCamera::shutdown() {
        cv::destroyWindow("Camera");
    }

}}
