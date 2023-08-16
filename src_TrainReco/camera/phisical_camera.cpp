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
        // Define the path to the directory containing the dataset images
        std::string datasetRootPath = "/home/eyelights/Documents/face_recognition/dataset/train-dataset/";

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

        // Read the dataset and labels
        std::vector<Mat> images;
        std::vector<int> index;

        // Initialize the label counter
        int labelCounter = 0;
        
        // Define the labels and corresponding person names
        std::vector<int> labels = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<cv::String> personNames = {
            "Frederic_Aubert", "Jean-Pierre_Stang", "Jules_Gregoire", "Neyl_Boukerche", "Pascal_Chevallier", "Alexandre_Goudeau", "Jerome_Pauc",
            "Ilia_Seliverstov", "Stephane_Grange", "Yannick_Durindel", "Nicolas_Hourcastagnou"
        };

        // Loop through the image paths
        for (int i = 0 ; i < 11 ; i++) { // Adjust the loop range based on the number of persons

            cv::String personName = personNames[i];
            std::cout << "Training for person: " << personName << std::endl;

            // Loop through the images for each person
            for (int j = 0 ; j < 500 ; j++) { // Adjust the loop range based on the number of images
                std::string imagePath = datasetRootPath + personName + "/" + personName + std::to_string(j) + ".jpg";
                Mat image = imread(imagePath, 1);

                if (image.empty()) {
                    std::cerr << "Failed to read image: " << imagePath << std::endl;
                    continue;
                }

                cvtColor(image, image, cv::COLOR_BGR2GRAY); // Convert to grayscale

                // Add the image and corresponding label to the vectors
                images.push_back(image);
                index.push_back(labelCounter);  // Assign incremental labels for training               
            }
            // Increment the label counter for the next person
            labelCounter++;
            
        }

        // Create and train the LBPHFaceRecognizer model
        cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create(1,10,10,10,80);
        model->train(images, index);

        // Create a mapping between labels and names
        std::unordered_map<int, std::string> labelNameMap;
        for (size_t i = 0; i < labels.size(); ++i) {
            labelNameMap[labels[i]] = personNames[i];
        }

        // Open the camera
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open the camera." << std::endl;
            return 1;
        }

        cv::Mat frame;
        cv::namedWindow("Face Recognition", cv::WINDOW_NORMAL);

        while (true) {
            // Read a frame from the camera
            cv::Mat frame;
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
                    // Resize the image to a consistent size (e.g., 100x100)
                    cv::Mat image;
                    cv::Mat faceROI;
                    resize(frame, image, Size(500, 500));
                    cvtColor(image, faceROI, cv::COLOR_BGR2GRAY); // Convert to grayscale

                    // Perform face recognition on the face region
                    int predictedLabel = -1;
                    double confidence = 0.0;
                    model->predict(faceROI, predictedLabel, confidence);

                    // Get the person name associated with the predicted label
                    std::string personName = labelNameMap[predictedLabel];

                    int x1 = static_cast<int>(detectionsMat.at<float>(i, 3) * frame.cols);
                    int y1 = static_cast<int>(detectionsMat.at<float>(i, 4) * frame.rows);
                    int x2 = static_cast<int>(detectionsMat.at<float>(i, 5) * frame.cols);
                    int y2 = static_cast<int>(detectionsMat.at<float>(i, 6) * frame.rows);
                    cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

                    // Display the predicted person name and confidence on the frame
                    std::string name = "Name: " + personName;
                    std::string conf = "Confidence: " + std::to_string(confidence);
                    cv::putText(frame, name, cv::Point(x1, y1), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
                    cv::putText(frame, conf, cv::Point(x1, y2), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 0), 2);
                }
            }

            // Display the result
            cv::imshow("Face Detection", frame);

            // Check for the 'q' key to exit the loop
            if (cv::waitKey(1) == 'q') {
                // Save the trained model
                std::string modelPath = "/home/eyelights/Documents/face_recognition/model.xml";
                model->save(modelPath);
                break;
            }
        }

        // Release the camera and close the window
        cap.release();
        cv::destroyAllWindows();
        
        return true;
    }

    void PhisicalCamera::_process(double delta) {
        if (isOpened) {
            cap >> currentFrame;
            if (!currentFrame.empty()) {
                imshow("Camera", currentFrame);
                waitKey(1);
            }
        }
    }

    void PhisicalCamera::shutdown() {
        destroyWindow("Camera");
    }

}}
