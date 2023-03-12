#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include <emscripten.h>


#include <chrono>
#include <iostream>
#include <vector>
#include <chrono>
#include <opencv2/imgproc.hpp>
#include <experimental_onnxruntime_cxx_api.h>


using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

int main(){}



void plot(){
    emscripten_run_script(" \\
        var img_array = new Uint8Array(document.rawDataSize); \\
        img_array.set(HEAPU8.subarray(document.dstPtr, document.dstPtr + document.rawDataSize)); \\
        var canvas = document.getElementById('viewport2'); \\
        var canvasCtx = canvas.getContext('2d'); \\
        var imageData = canvasCtx.createImageData(document.w, document.h); \\
        imageData.data.set(img_array); \\
        canvasCtx.putImageData(imageData, 0, 0); \\
    ");
}

int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= i;
  return total;
}


double minVal; 
double maxVal; 
cv::Point minLoc; 
cv::Point maxLoc;
int netInputChannels = 1;
int netInputWidth = 64;
int netInputHeight = 64;
Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");

class ModelWrapper{
    private:
        std::string modelName = "./model.onnx";
        Ort::SessionOptions options;
        Ort::Experimental::Session session;
    
    public:

        ModelWrapper():
            options(Ort::SessionOptions()),
            session(env, modelName, options){
            std::cout << "Model was initialized." << std::endl;
        };

        // void detect_heart_batch(std::vector<float> preparedVect){
        //     int batch_num = preparedVect.size()/netInputChannels/netInputWidth/netInputHeight;

        //     std::vector<int64_t> input_shape(4);
        //     input_shape[0] = batch_num; // batch num
        //     input_shape[1] = 1;
        //     input_shape[2] = netInputWidth;
        //     input_shape[3] = netInputHeight;

        //     std::vector<float> input_tensor_values(calculate_product(input_shape));
        //     for(int i=0; i<input_shape[0]; i++){
        //         std::copy_n(preparedVect.begin(), preparedVect.size(), &input_tensor_values[i*preparedVect.size()]);
        //     }
        //     std::vector<Ort::Value> input_tensors;
        //     input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(input_tensor_values.data(), input_tensor_values.size(), input_shape));
        //     auto output_tensors = session.Run(session.GetInputNames(), input_tensors, session.GetOutputNames());

        //     auto& tensor = output_tensors[0];
        //     float* tensorData = tensor.GetTensorMutableData<float>();
        //     auto out_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();

        //     for(int i=0; i<batch_num; i++){
        //         cv::Mat result = cv::Mat(out_shape[2], out_shape[3], CV_32FC1, tensorData+i*out_shape[2]*out_shape[3]);
        //         minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc );
        //         std::cout<<minVal<<" "<<maxVal<<std::endl;

        //         cv::Mat thresh;
        //         cv::threshold(result, thresh, 0, 255, cv::THRESH_BINARY);
        //         thresh.convertTo(thresh, CV_8UC1);
        //         std::vector<std::vector<cv::Point>> contours;
        //         std::vector<cv::Vec4i> hierarchy;
        //         cv::findContours(thresh, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        //         if(contours.size()==0) continue;

        //         float max_area = -1;
        //         int max_cnt = -1;
        //         for(int cnt_i=0; cnt_i<contours.size(); cnt_i++){
        //         float area = cv::contourArea(contours[cnt_i]);
        //         if(area>max_area){
        //             max_area = area;
        //             max_cnt = cnt_i;
        //         }      
        //         }
        //         if(max_area<200) continue;
                
        //         cv::Rect rect = cv::boundingRect(contours[max_cnt]);
        //         std::cout<<rect.x<<" "<<rect.y<<" "<<rect.height<<" "<<rect.width<<std::endl;
        //         cv::rectangle(result, rect, 255);
        //         cv::imwrite("../out_"+to_string(i)+".png", result);
        //     }

        //     // return dets;
        // };

        int plotDetection(int n, int src_size, uintptr_t srcPtr, uintptr_t detectionsPtr, uintptr_t dstPtr){
            uint8_t* srcPtr_ = reinterpret_cast<uint8_t*>(srcPtr);
            int* detectionsPtr_ = reinterpret_cast<int*>(detectionsPtr);
            uint8_t* dstPtr_ = reinterpret_cast<uint8_t*>(dstPtr);

            auto t1 = high_resolution_clock::now();
            for(int i=0; i<n; i++){
                cv::Mat image = cv::Mat(src_size, src_size, CV_8UC4, srcPtr_+i*4*src_size*src_size);
                if(*(detectionsPtr_+i*5) == 1){
                    int x = *(detectionsPtr_+i*5+1); // x
                    int y = *(detectionsPtr_+i*5+2); // y
                    int h = *(detectionsPtr_+i*5+3); // w
                    int w = *(detectionsPtr_+i*5+4); // h
                    cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(255, 255, 255, 255));
                }
                uint8_t* resPtr_ = reinterpret_cast<uint8_t*>(image.data);
                std::memcpy(dstPtr_+i*4*src_size*src_size, resPtr_, 4*src_size*src_size);
            }
            auto t2 = high_resolution_clock::now();
            return duration_cast<milliseconds>(t2 - t1).count();
        }
        
        int resize(int n, int src_size, int dst_size, uintptr_t srcPtr, uintptr_t dstPtr){
            uint8_t* srcPtr_ = reinterpret_cast<uint8_t*>(srcPtr);
            uint8_t* dstPtr_ = reinterpret_cast<uint8_t*>(dstPtr);
            
            auto t1 = high_resolution_clock::now();
            for(int i=0; i<n; i++){
                cv::Mat original = cv::Mat(src_size, src_size, CV_8UC4, srcPtr_+i*4*src_size*src_size);
                // cv::cvtColor(original, original, cv::COLOR_RGBA2GRAY);
                cv::Mat resized;
                cv::resize(original, resized, cv::Size(dst_size, dst_size));
                // cv::cvtColor(resized, resized, cv::COLOR_GRAY2RGBA);
                uint8_t* resPtr_ = reinterpret_cast<uint8_t*>(resized.data);
                std::memcpy(dstPtr_+i*4*dst_size*dst_size, resPtr_, 4*dst_size*dst_size);

                // for(int i=0; i<100; i++)
                //     std::cout<<(float)*(resized.data+i)<<std::endl;
            }
            auto t2 = high_resolution_clock::now();
            return duration_cast<milliseconds>(t2 - t1).count();
        }

        int preprocess(int n, int src_size, int dst_size, uintptr_t srcPtr, uintptr_t dstPtr){
            uint8_t* srcPtr_ = reinterpret_cast<uint8_t*>(srcPtr);
            // uint8_t* dstPtr_ = reinterpret_cast<uint8_t*>(dstPtr);
            float* dstPtr_ = reinterpret_cast<float*>(dstPtr);
            
            auto t1 = high_resolution_clock::now();
            for(int i=0; i<n; i++){
                cv::Mat img = cv::Mat(src_size, src_size, CV_8UC4, srcPtr_+i*4*src_size*src_size);
                cv::cvtColor(img, img, cv::COLOR_RGBA2GRAY);
                cv::resize(img, img, cv::Size(dst_size, dst_size));
                img.convertTo(img, CV_32FC1);

                float* resPtr_ = reinterpret_cast<float*>(img.data);
                std::memcpy(dstPtr_+i*dst_size*dst_size, resPtr_, 4*dst_size*dst_size);

                // for(int i=0; i<100; i++)
                //     std::cout<<(float)*(dstPtr_+i)<<std::endl;
                    // std::cout<<(float)*(img.data+i)<<std::endl;
            }
            auto t2 = high_resolution_clock::now();
            return duration_cast<milliseconds>(t2 - t1).count();
        }


        int detect(int n, uintptr_t srcPtr, uintptr_t dstPtr){
            // args:
            //      srcPtr - image to detect heart
            //      dstPtr - for results [height, width, top, left] - 4 ints fo most confident detected object
            //
            // returns: 
            //      float - confidence>0 or 0 if obejct wasn't found

            float* srcPtr_ = reinterpret_cast<float*>(srcPtr);
            int* dstPtr_ = reinterpret_cast<int*>(dstPtr);

            auto t1 = high_resolution_clock::now();

            int img_size = 64;
            std::vector<int64_t> input_shape(4);
            input_shape[0] = n;
            input_shape[1] = 1;
            input_shape[2] = img_size;
            input_shape[3] = img_size;
            std::vector<float> input_tensor_values(calculate_product(input_shape));
            input_tensor_values.assign(srcPtr_, srcPtr_+n*img_size*img_size);

            std::vector<Ort::Value> input_tensors;
            input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(input_tensor_values.data(), input_tensor_values.size(), input_shape));
            auto output_tensors = session.Run(session.GetInputNames(), input_tensors, session.GetOutputNames());
            auto& tensor = output_tensors[0];
            float* tensorData = tensor.GetTensorMutableData<float>();
            auto out_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();


            for(int i=0; i<n; i++){
                cv::Mat result = cv::Mat(out_shape[2], out_shape[3], CV_32FC1, tensorData+i*out_shape[2]*out_shape[3]);
                cv::Mat thresh;
                cv::threshold(result, thresh, 0, 255, cv::THRESH_BINARY);
                thresh.convertTo(thresh, CV_8UC1);
                std::vector<std::vector<cv::Point>> contours;
                std::vector<cv::Vec4i> hierarchy;
                cv::findContours(thresh, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
                if(contours.size()==0){
                    *(dstPtr_+i*5) = 0;
                    continue;
                }

                float max_area = -1;
                int max_cnt = -1;
                for(int cnt_i=0; cnt_i<contours.size(); cnt_i++){
                float area = cv::contourArea(contours[cnt_i]);
                if(area>max_area){
                    max_area = area;
                    max_cnt = cnt_i;
                }      
                }
                if(max_area<200){
                    *(dstPtr_+i*5) = 0;
                    continue;
                }
                
                cv::Rect rect = cv::boundingRect(contours[max_cnt]);
                *(dstPtr_+i*5) = 1;
                *(dstPtr_+i*5+1) = (int)(1.0*rect.x/img_size*512); // x
                *(dstPtr_+i*5+2) = (int)(1.0*rect.y/img_size*512); // y
                *(dstPtr_+i*5+3) = (int)(1.0*rect.height/img_size*512); // w
                *(dstPtr_+i*5+4) = (int)(1.0*rect.width/img_size*512); // h
                // std::cout<<"detected "<<i<<std::endl;
            }

            // uint8_t* dstPtr_ = reinterpret_cast<uint8_t*>(dstPtr);
            // *(dstPtr_) = 128; // ok

            // int* dstPtr_ = reinterpret_cast<int*>(dstPtr);
            // *(dstPtr_) = 128;

            // float* dstPtr_ = reinterpret_cast<float*>(dstPtr);
            // *(dstPtr_) = 1024.5; // ok

            auto t2 = high_resolution_clock::now();
            return duration_cast<milliseconds>(t2 - t1).count();
        }
};


int run_batch_test(int batch_size, int num_batches){
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "batch-model-explorer");
    Ort::SessionOptions session_options;
    std::string modelName = "./model.onnx";
    Ort::Experimental::Session session = Ort::Experimental::Session(env, modelName, session_options);

    auto input_shapes = session.GetInputShapes();
    auto input_names = session.GetInputNames();
    auto output_names = session.GetOutputNames();

    // int batch_size = n;
    // int num_batches = 50;
    auto input_shape = input_shapes[0];
    // assert(input_shape[0] == -1);  // symbolic dimensions are represented by a -1 value
    input_shape[0] = batch_size;
    int num_elements_per_batch = calculate_product(input_shape);

    // Create an Ort tensor containing random numbers
    std::vector<float> batch_input_tensor_values(num_elements_per_batch);
    std::generate(batch_input_tensor_values.begin(), batch_input_tensor_values.end(), [&] { return rand() % 255; });  // generate random numbers in the range [0, 255]
    std::vector<Ort::Value> batch_input_tensors;
    batch_input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(batch_input_tensor_values.data(), batch_input_tensor_values.size(), input_shape));

    auto t1 = high_resolution_clock::now();
    // process multiple batches
    for (int i = 0; i < num_batches; i++) {
        // std::cout << "\nProcessing batch #" << i << std::endl;
        // pass data through model
        try {
            auto batch_output_tensors = session.Run(input_names, batch_input_tensors, output_names);
        } catch (const Ort::Exception& exception) {
            // std::cout << "ERROR running model inference: " << exception.what() << std::endl;
            std::cout << "ERROR running model inference" << std::endl;
            return -1;
        }
    }

    auto t2 = high_resolution_clock::now();
    return int(1.0*duration_cast<milliseconds>(t2 - t1).count()/num_batches);
}

EMSCRIPTEN_BINDINGS(DETECTION_DEMO){
    emscripten::class_<ModelWrapper>("ModelWrapper")
        .constructor<>()
        .function("plotDetection", &ModelWrapper::plotDetection)
        .function("preprocess", &ModelWrapper::preprocess)
        .function("resize", &ModelWrapper::resize)
        .function("detect", &ModelWrapper::detect);
    emscripten::function("run_batch_test", &run_batch_test);
}