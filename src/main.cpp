#include "graph.h"
#include "colorLines.h"
// #include <opencv2/opencv.hpp>

// #define K 1e4
#define K 5e2

using namespace std;
using namespace cv;

typedef Graph<float,float,float> GraphType;

int oneD(int x, int y , int rows){
    return x*rows + y;
}

int main(int argc, char const *argv[])
{
    cv::Mat image = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    cv::Size originalSize = image.size();

    Mat image_segnet = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    cv::resize(image,image,image_segnet.size());

    Mat image_road = imread(argv[3], CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(image_road, image_road, cv::COLOR_BGR2GRAY);
    
    Mat image_nonRoad = imread(argv[4], CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(image_nonRoad, image_nonRoad, cv::COLOR_BGR2GRAY);
    
    const int N = image.rows*image.cols;
    Vec3b roadColor = Vec3b(4,4,4);

    GraphType *g = new GraphType(/*estimated # of nodes*/ N, /*estimated # of edges*/ N);
    for (int i = 0; i < N; ++i){
        g->add_node();
    }

    colorLines c;
    c.init(image,10);
    std::vector<float> lineProbsRoad , lineProbsNonRoad ;
    lineProbsRoad.resize(c.lines.size());
    lineProbsNonRoad.resize(c.lines.size());

    for (int i = 0; i < lineProbsRoad.size(); ++i){
        lineProbsRoad[i] = 0;
        lineProbsNonRoad[i] = 0;
    }

    // cout<<c.lines.size();

    for (int i = 0; i < image.cols; ++i){
        for (int j = 0; j < image.rows; ++j){
                
                float p_r = (float)image_road.at<uchar>(j,i); 
                float p_nr = (float)image_nonRoad.at<uchar>(j,i); 

                Point3d pixel = (Point3d)image.at<Vec3b>(j,i);
                std::vector<float> temp = c.get_probability(pixel);

                int maxPos = 0;
                float maxVal = 0;

                for (int k = 0; k < temp.size(); ++k){
                     if (temp[k] > maxVal){
                         maxVal = temp[k];
                         maxPos = k;
                     }
                } 
            
                if( p_r > p_nr)
                    lineProbsRoad[maxPos] += 1;
                else
                    lineProbsNonRoad[maxPos] += 1;
        }
    }

    int roadLine = 0;
    float maxVal = 0;

    for (int k = 0; k < lineProbsRoad.size(); ++k){
        if (lineProbsRoad[k] > maxVal){
             maxVal = lineProbsRoad[k];
            roadLine = k;
        }
    } 


    for (int i = 0; i < image.cols; ++i){
        for (int j = 0; j < image.rows; ++j){
            float p_r = (float)image_road.at<uchar>(j,i); 
            float p_nr = (float)image_nonRoad.at<uchar>(j,i); 
            
            int nodeId = oneD(i,j,image.rows);
            float normalize = p_r + p_nr;
            p_r /= normalize;
            p_nr /= normalize;
            // g -> add_tweights( nodeId , p_r , p_nr );

            ///////
            Point3d pixel = (Point3d)image.at<Vec3b>(j,i);
            std::vector<float> temp = c.get_probability(pixel);

            int maxPos = 0;
            float maxVal = 0;

            for (int k = 0; k < temp.size(); ++k){
                 if (temp[k] > maxVal){
                     maxVal = temp[k];
                     maxPos = k;
                 }
            } 
        
            if( maxPos == roadLine){
                p_r *= 0.7;
                p_nr *= 0.3;
                p_r /= normalize;
                p_nr /= normalize;

                g -> add_tweights( nodeId , p_r , p_nr );
            }
            else
                g -> add_tweights( nodeId , p_r , p_nr );
            ////////

        }
    }

    int nbx[4] = {-1 , 1, 0 , 0 }; 
    int nby[4] = {0 , 0 , -1 ,1 }; 

    for (int i = 1; i < image.cols-1; i+=1){
        for (int j = 1; j < image.rows-1; j+=1){
            
            Point3d pixel1 = (Point3d)image.at<Vec3b>(j,i);  
            
            for (int k = 0; k < 4; ++k){
                Point3d pixel2 = (Point3d)image.at<Vec3b>( j+nby[k] , i+nbx[k] );
                float prob = c.get_probability2(pixel1,pixel2);
                cout<<"asdasdas : "<<prob<<endl;
                g -> add_edge(  oneD(i,j,image.rows) , oneD(i+nbx[k],j+nby[k],image.rows) , K*prob , K*prob );

                // cout<<prob<<endl;
            }
        
        }
    }

    float flow = g -> maxflow();
    printf("Flow = %f\n", flow);

    Mat result = image.clone();

    for (int i = 0; i < image.cols; ++i){
        for (int j = 0; j < image.rows; ++j){
            
            int n1;
            n1 = oneD(i,j,image.rows);
            
            if (g->what_segment(n1) == GraphType::SOURCE){
                    result.at<Vec3b>(j,i) = Vec3b(255,105,180);
                }
            else{
                    result.at<Vec3b>(j,i) = Vec3b(0,0,0);
                }// cout<<"SINK"<<endl;
            
        }
    }
    // imwrite( "data/segmentation_wi.png", image );
    Mat dst;
    float alpha , beta;
    alpha = 0.8;
    // beta = ( 1.0 - alpha );
    beta = 0.6;
    // cv::resize(image,image,img.size());

    addWeighted( image, alpha, result, beta, 0.0, dst);
    imwrite( "Result.png", dst );
    cout<<"Lines : "<<c.lines.size()<<endl;
  
    
    for (int i = 0; i < lineProbsRoad.size(); ++i){
        cout<<i<<"   "<<lineProbsRoad[i]<<"   "<<lineProbsNonRoad[i]<<endl;
    }

    return 0;
}