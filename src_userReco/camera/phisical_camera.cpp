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

    double PhisicalCamera::delta(Mat& img, int i, int j, int k, int l) {
        Vec3b pixel1 = img.at<Vec3b>(i, j);
        Vec3b pixel2 = img.at<Vec3b>(k, l);
        
        double x = sqrt(pixel1[0] * pixel1[0] + pixel1[1] * pixel1[1] + pixel1[2] * pixel1[2]);
        double y = sqrt(pixel2[0] * pixel2[0] + pixel2[1] * pixel2[1] + pixel2[2] * pixel2[2]);
        
        return x - y;
    }

    int PhisicalCamera::vectorValue(Mat& img, int i, int j) {
        double verticale = abs(delta(img, i, j - 1, i, j + 1)) / 256.0;
        double horizontale = abs(delta(img, i - 1, j, i + 1, j)) / 256.0;
        double diagonale1 = abs(delta(img, i - 1, j - 1, i + 1, j + 1)) / 256.0;
        double diagonale2 = abs(delta(img, i - 1, j + 1, i + 1, j - 1)) / 256.0;
        
        return static_cast<int>((horizontale * 90 + diagonale1 * 180 + diagonale2 * 270) / 4);
    }

    bool PhisicalCamera::open(int cameraId) {

        // Load the trained model
        cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create();
        model->read("/home/eyelights/Documents/face_recognition/model.xml");

        // Define the labels and corresponding person names
        std::vector<std::vector<cv::String>> user = {{"Frederic_Aubert","Sport"}, {"Jean-Pierre_Stang","Comfort"}, {"Jules_Gregoire","Eco"},
            {"Neyl_Boukerche","Maximum Range"}, {"Pascal_Chevallier","Sport"}, {"Alexandre_Goudeau","Comfort"}, {"Jerome_Pauc","Eco"},
            {"Ilia_Seliverstov","Maximum Range"}, {"Stephane_Grange","Sport"}, {"Yannick_Durindel","Comfort"}, {"Nicolas_Hourcastagnou","Eco"} };

        // Load pre-trained model
        std::string modelFile = "/home/eyelights/Documents/face_detection/detect_godot/src/camera/res10_300x300_ssd_iter_140000.caffemodel";
        std::string configFile = "/home/eyelights/Documents/face_detection/detect_godot/src/camera/deploy.prototxt";
        cv::dnn::Net net = cv::dnn::readNetFromCaffe(configFile, modelFile);
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

        // Check if the model was loaded successfully
        if (net.empty()) {
            std::cout << "Error loading the model." << std::endl;
            return -1;
        }

        // Open the camera
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open the camera." << std::endl;
            return 1;
        }

        cv::Mat frame;
        cv::Mat image;
        cv::Mat faceROI;
        cv::Mat Face;
        cv::Mat face;
        cv::String mode;

        while (true) {
            // Read a frame from the camera
            cap.read(frame);
            
            // Perform face detection
            cv::Mat blob = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104, 177, 123));
            net.setInput(blob);
            cv::Mat detections = net.forward();

            // Process the detections and draw bounding boxes around faces
            cv::Mat detectionsMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

            for (int i = 0; i < detectionsMat.rows; ++i) {
                float confidence = detectionsMat.at<float>(i, 2);

                if (confidence > 0.3) {  // You can adjust this threshold as needed
                    int x1 = static_cast<int>(detectionsMat.at<float>(i, 3) * frame.cols);
                    int y1 = static_cast<int>(detectionsMat.at<float>(i, 4) * frame.rows);
                    int x2 = static_cast<int>(detectionsMat.at<float>(i, 5) * frame.cols);
                    int y2 = static_cast<int>(detectionsMat.at<float>(i, 6) * frame.rows);

                    cv::Rect rec(x1, y1, x2 - x1, y2 - y1);
                    Face=frame(rec); // Slicing to crop the image
                    Mat newImg = Face.clone();
                                for (int i = 1; i < Face.rows - 1; ++i) {
                                    for (int j = 1; j < Face.cols - 1; ++j) {
                                        int vectorVal = vectorValue(Face, i, j);
                                        newImg.at<Vec3b>(i, j) = Vec3b(vectorVal, vectorVal, vectorVal);
                                    }
                                }
                    cvtColor(newImg, face, cv::COLOR_BGR2GRAY); // Convert to grayscale

                    // Perform face recognition on the face region
                    int predictedLabel = -1;
                    double Confidence = 0.0;
                    model->predict(face, predictedLabel, Confidence);
                    
                    // Display the predicted person name and confidence
                    if (predictedLabel!=-1) {
                        std::cout << "Name: " << user[predictedLabel][0] << std::endl;
                        if (mode!=user[predictedLabel][1]) {
                            cv::String mode = user[predictedLabel][1];
                            std::cout << "Switching to mode " << mode << std::endl;
                        }
                        std::cout << "Confidence: " << std::to_string(int(Confidence*100)/100) << "%" << std::endl;
                    }
                else {
                    std::cout << "Unknown user. Want to register as a user of this car ?" << std::endl;
                }
                }
            }
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