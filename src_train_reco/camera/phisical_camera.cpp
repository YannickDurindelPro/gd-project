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
                Mat image = imread(imagePath, IMREAD_GRAYSCALE);
                image.resize(400, 400 * image.rows / image.cols);

                if (image.empty()) {
                    std::cerr << "Failed to read image: " << imagePath << std::endl;
                    continue;
                }

                // Add the image and corresponding label to the vectors
                images.push_back(image);
                index.push_back(labelCounter);  // Assign incremental labels for training               
            }
            // Increment the label counter for the next person
            labelCounter++;
            
        }
        std::cout << "images loaded" << std::endl;

        // Create and train the LBPHFaceRecognizer model
        cv::Ptr<cv::face::LBPHFaceRecognizer> model = cv::face::LBPHFaceRecognizer::create(1,9,8,8,100);
        std::cout << "model created" << std::endl;
        model->train(images, index);
        std::cout << "model trained" << std::endl;

        // Save the trained model
        std::string modelPath = "/home/eyelights/Documents/face_recognition/model.xml";
        std::cout << "model path defined" << std::endl;
        model->save(modelPath);
        std::cout << "model saved" << std::endl;
        
        abort();
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
