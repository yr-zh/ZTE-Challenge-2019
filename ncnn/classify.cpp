//#include <io.h>
#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <algorithm>
#include <vector>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

#if NCNN_VULKAN
#include "gpu.h"
#endif // NCNN_VULKAN
#include "mat.h"
#include "net.h"

using namespace std;
class ZTE_AI{
    public:
    ncnn::Net net;
    
//     void classify(cv::Mat img, std::vector<float>& scores, ncnn::Extractor extractor);
    vector<float> get_feature(string image_name);
};

vector<float> ZTE_AI::get_feature(string image_name){

#if NCNN_VULKAN
    ncnn::create_gpu_instance();
#endif // NCNN_VULKAN
    
    ncnn::Extractor extractor = net.create_extractor();
    extractor.set_light_mode(true);
    extractor.set_num_threads(4);
    
    // detect one image and output 256 vector
    string image_path = "/home/ubuntu/MyFiles/ZTE/FACE-ALL-5-POINTS_CROP/";
    string image = image_path + image_name;
    cout<<image<<endl;

    cv::Mat img = cv::imread(image, 1);
//     unsigned char* rgbdata;// data pointer to RGB image pixels
    int w=128;// image width
    int h=128;// image height
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows, w, h );
    
    std::vector<float> scores;
    const float mean_vals[3] = {127.5, 127.5, 127.5};
    const float norm[3] = {0.0078125f, 0.0078125f, 0.0078125f};
    
//     for(int i=0;i<img.rows;i++)
//       {
//           for(int j=0;j<img.cols;j++)
//           {
//               img.at<cv::Vec3b>(i,j)[0]=int(float((img.at<cv::Vec3b>(i,j))[0]-mean1)*norm);
//               img.at<cv::Vec3b>(i,j)[1]=int(float((img.at<cv::Vec3b>(i,j))[1]-mean1)*norm);
//               img.at<cv::Vec3b>(i,j)[2]=int(float((img.at<cv::Vec3b>(i,j))[2]-mean1)*norm);
//          }
//     }
    
    // 图片预处理
    input.substract_mean_normalize(mean_vals,0);
    input.substract_mean_normalize(0,norm);
    
    // ncnn前向计算
    extractor.input("data", input);
    ncnn::Mat output;
    extractor.extract("fc5", output);

    // 输出预测结果
    ncnn::Mat out_flatterned = output.reshape(output.w * output.h * output.c);
    scores.resize(out_flatterned.w);

    for (int j=0; j<out_flatterned.w; j++)
    {
        scores[j] = out_flatterned[j];
    }
    
    return scores;
};

float CosineVal(vector<float>& feature1, vector<float>& feature2)
{
    assert(feature1.size() == feature2.size());
    float sumAA = 0;
    float sumBB = 0;
    float sumAB = 0;
    for(int i=0; i<feature1.size(); ++i)
    {
        sumAA += feature1[i] * feature1[i];
        sumBB += feature2[i] * feature2[i];
        sumAB += feature1[i] * feature2[i];
    }
    
    float res = sumAB / (sqrt(sumAA*sumBB) + 1e-5);
    return res;
};

void GetTPR(ZTE_AI& ai, vector<string>& images_list1, vector<string>& images_list2)
{

    vector<vector<float>> features1, features2;
    assert(images_list1.size() == images_list2.size());
    
    for(int i=0; i<images_list1.size(); ++i)
    {
        vector<float> tmp_feature1 = ai.get_feature(images_list1[i]);
        features1.push_back(tmp_feature1);
        vector<float> tmp_feature2 = ai.get_feature(images_list2[i]);
        features2.push_back(tmp_feature2);
    }
    
    cout<<"feature has extracted"<<endl;
    
    vector<float> posSims;
    vector<float> negSims;
    
    for(int i=0; i<features1.size(); ++i)
    {
        for(int j=0; j<features2.size(); ++j)
        {
            float sim = CosineVal(features1[i],features2[j]);
            if(i == j)
            {
                posSims.push_back(sim);
            }
            else
            {
                negSims.push_back(sim);
            }
        }
    }
    
    sort(negSims.begin(),negSims.end(),greater<float>());
    sort(posSims.begin(),posSims.end(),less<float>());
    
    cout<<negSims.size()*0.0001<<endl;
    float threshold_00001=negSims[negSims.size()*0.0001];
    int posErrorNum_00001=0;
    
    
    
    cout<<"negSims: "<<endl;
    for(int i=0; i<2; i++){
//         cout<<negSims[i]<<endl;
//         cout<<negSims[1000-i]<<endl;
        for(int j=0; j<128; j++){
            cout<<features1[i][j]<<" ";
            
        }
        cout<< " "<<endl;
         cout<< " "<<endl;
         cout<< " "<<endl;
    }
    
    
    cout<<"negSims: "<<endl;
    for(int i=0; i<2; i++){
//         cout<<negSims[i]<<endl;
//         cout<<negSims[1000-i]<<endl;
        for(int j=0; j<128; j++){
            cout<<features2[i][j]<<" ";
            
        }
        cout<< " "<<endl;
         cout<< " "<<endl;
         cout<< " "<<endl;
    }
    cout<<"posSims: "<<endl;
    for(int i=0; i<10; i++){
//         cout<<posSims[100-i]<<endl;
//         cout<<posSims[i]<<endl;

    }
    
    for(int i=0;i<posSims.size();i++)
    {
        if(posSims[i]<threshold_00001)
        {
            posErrorNum_00001++;
           
        }
        else
            break;
    }
    cout<<"posSimsError: "<<endl;
    cout<<posErrorNum_00001<<endl;
        
    float TPR_00001=(posSims.size()-posErrorNum_00001)/(float)posSims.size();
    
    cout << "TPR is " << TPR_00001 << " @FPR=0.0001."<<endl;
}

void read_text_image(string image_path, string test_txt, vector<string> &image1_list, vector<string> &image2_list){
    
    ifstream file(test_txt);

    string line = "";       // each line
    string image1 = "";   // image1
    string image2 = "";      //image2
    
    // get image1_list and image2_list
    int num = 1000;
    for(int i=0; i<num; i++){
        
        getline(file, line);
        stringstream word(line);
        word >> image1;
        word >> image2;
        image1_list.push_back(image1);
        image2_list.push_back(image2);
//         cout<<line<<endl;
//         cout<<image1<<endl;
//         cout<<image2<<endl;
    }
    
}

int main(){
    ZTE_AI ai;
    
//     ncnn::Net net;
    ai.net.load_param("./zte.v0.param");
    ai.net.load_model("./zte.v0.bin");

#if NCNN_VULKAN
    ai.net.use_vulkan_compute = true;
#endif // NCNN_VULKAN

    // net.use_int8_inference = 0;
    // get the test image
    string image_path = "/home/ubuntu/MyFiles/ZTE/FACE-ALL-5-POINTS_CROP/";
    string test_txt = "/home/ubuntu/MyFiles/ZTE/test_2000_images_list.txt";
    vector<string> image1_list;
    vector<string> image2_list;
    read_text_image(image_path, test_txt, image1_list, image2_list);
    GetTPR(ai, image1_list, image2_list);
    // vector<float> feature;
    // feature = ai.ZTE_AI::get_feature();
#if NCNN_VULKAN
    ncnn::destroy_gpu_instance(ai.net);
#endif // NCNN_VULKAN
    return 0;
}

