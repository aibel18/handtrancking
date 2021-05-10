/**
 *===================================================================================
 *                 @name stereovision.h
 *               @author abel ticona
 *                @email jaticona@gmail.com
 *              @version 0.1
 *===================================================================================
**/

#ifndef STEREOVISION_H_INCLUDED
#define STEREOVISION_H_INCLUDED


#include <opencv2/opencv.hpp>
#include <iostream>
#include <time.h>
#include <cstdio>
#include <thread>

using namespace std;
using namespace cv;

/**
 *-----------------------------------------------------------------------------------
 *           Name: stereo vision
 *  Last Modified: , 2017
 *    Description: class that implement algorithm for stereo matching, filter average,
                    and bilateral
 *-----------------------------------------------------------------------------------
**/

/**
 * Function cost for Sum of Absolute Differences
 */
float funSAD(float d){
    return abs(d);
}
/**
 * Function cost for Sum of Squared Differences
 */
float funSSD(float d){
    return d*d;
}
/**
 * Function cost for Sum of Squared Differences robust
 */
float funSSDR(float d){
    float dd = d*d;
    return dd/(dd + 100 );
}

/// pointer to a function cost (default SSD)
float (*funCost)(float) = funSSD;
/// sigma to create filter Gaussian (default 0.6)
float sigma = 0.6;
/// pointer to a array filter
float *filter;

/**
 * Function that create a filter weighted average
 */
void createFilterAverage(int sizeBlock){

    /// size filter
    int n = sizeBlock*2 + 1;

    if(!filter)
        filter = new float [n*n];

    float total = 0.0f;

    for(int i=-sizeBlock,k=0;i<=sizeBlock;i++){
        for(int j=-sizeBlock;j<=sizeBlock;j++,k++){
            filter[k] = n - (abs(i) + abs(j));
            total += filter[k];
        }
    }
    for(int i=-sizeBlock,k=0;i<=sizeBlock;i++){
        for(int j=-sizeBlock;j<=sizeBlock;j++,k++){
            filter[k] = filter[k] / total;
        }
    }
}
/**
 * Function that create a filter Gaussian ( for filter bilateral)
 */
void createFilterGaussian(int sizeBlock){

    /// size filter
    int n = sizeBlock*2 + 1;

    if(!filter)
        filter = new float [n*n];

    for(int i=-sizeBlock,k=0;i<=sizeBlock;i++){
        for(int j=-sizeBlock;j<=sizeBlock;j++,k++){
            filter[k] = 1/(2*3.141*sigma*sigma)*exp( - ( ( i*i + j*j )/(2*sigma*sigma) ) );
        }
    }
}
/**
 * Function for calculate the difference between intensities ( for filter bilateral)
 */
void getBlockDifferenceIntensity(Mat *image,int i,int j,unsigned char *block,int sizeBlock){

    int lengthImage = image->step;

    int n = 0;

    int indexFixe = i * lengthImage + j*3;

    for(int ii= -sizeBlock;ii<=sizeBlock;ii++){
        for(int jj=-sizeBlock;jj<=sizeBlock;jj++){
            int index = ( i + ii) * lengthImage + (j + jj)*3;


            /// calculate the cost for each channel
            float costR = (float)image->data[ indexFixe ] - (float)image->data[ index ];
            float costG = (float)image->data[ indexFixe +1] - (float)image->data[ index + 1];
            float costB = (float)image->data[ indexFixe +2] - (float)image->data[ index + 2];

            /// cost total the pixel
            /// distance euclidean  d = sqrt( R*R + G*G + B*B )
            block[n] = costR*costR + costG*costG + costB*costB;

            /// exp( - d*d/ (2*sigma*sigma) )
            /// simplifying the equation => exp( - R*R + G*G + B*B + / (2*sigma*sigma) )
            block[n++] = exp( - ( block[n] /(2*sigma*sigma) ) );

        }
    }
}


/**
 * Function for calculate cost between the blocks
 */
float calculateCost(unsigned char *blockL,unsigned char *blockR,int lengthBlock){

    float cost = 0;
    float diff = 0.0f;

    for(int k=0;k<lengthBlock;k++){
        diff = blockR[k] - blockL[k];
        cost += funCost(diff);
    }
    return cost;
}

/**
 * Function for calculate cost for block aggregation cost the neighbors, without filter
 */
float calculateAggregationCost(Mat*image,float **imageCost,int i,int j,int sizeBlock){

    float cost = 0;
    int f = 0;
    for(int ii= -sizeBlock;ii<=sizeBlock;ii++){
        for(int jj=-sizeBlock;jj<=sizeBlock;jj++,f++){
            cost+=imageCost[i+ii][(j+jj)];
        }
    }

    return cost;
}
/**
 * Function for calculate cost for block aggregation cost the neighbors, with filter average
 */
float calculateAggregationCostFilterAverage(Mat*image,float **imageCost,int i,int j,int sizeBlock){

    float cost = 0;
    int f = 0;
    for(int ii= -sizeBlock;ii<=sizeBlock;ii++){
        for(int jj=-sizeBlock;jj<=sizeBlock;jj++,f++){
            cost+= filter[f] * imageCost[i+ii][(j+jj)];
        }
    }
    return cost;
}
/**
 * Function for calculate cost for block aggregation cost the neighbors, with filter bilateral
 */
float calculateAggregationCostFilterBilateral(Mat*image,float **imageCost,int i,int j,int sizeBlock){

    int n = sizeBlock*2 + 1;
    unsigned char *filterIntensity = new unsigned char [n*n];

    /// difference between intensity
    getBlockDifferenceIntensity(image, i, j,filterIntensity, sizeBlock);

    float cost = 0;
    int f = 0;
    for(int ii= -sizeBlock;ii<=sizeBlock;ii++){
        for(int jj=-sizeBlock;jj<=sizeBlock;jj++,f++){
            cost+=filter[f] * filterIntensity[f] * imageCost[i+ii][(j+jj)];
        }
    }
    delete[] filterIntensity;

    return cost;
}

/// pointer to a function cost (default without filter)
float (*funAggregation)(Mat*image,float **,int ,int ,int ) = calculateAggregationCost;



/**
 * Function that return the block on the coordinate i,j
 */
void getBlock(Mat *image,int i,int j,unsigned char *block,int sizeBlock){

    int lengthImage = image->step;

    int n = 0;

    for(int ii= -sizeBlock;ii<=sizeBlock;ii++){
        for(int jj=-sizeBlock;jj<=sizeBlock;jj++){
            int index = ( i + ii) * lengthImage + (j + jj)*3;
            block[n++] =(int)image->data[ index ];
            block[n++] = (int)image->data[ index + 1 ];
            block[n++] = (int)image->data[ index + 2 ];
        }
    }
}

/**
 * Function that disparity map (force gross)
 */
void disparity( Mat& imageL,Mat& imageR,Mat& result,int rangeDisparity,int sizeBlock){

    result = Mat(imageR.size(), CV_8UC1, cv::Scalar(0));

    int lengthBlock = (sizeBlock*2+1)*(sizeBlock*2+1)*3;

    unsigned char *blockR = new unsigned char [lengthBlock];
    unsigned char *blockL = new unsigned char [lengthBlock];

    float costMin = 1000000;
    int disparity = 0;
    float costCurrent = 0.0;

    /// factor of normalization
    float normalize = 255.0f/rangeDisparity;

    for(int i=sizeBlock;i<imageR.rows-sizeBlock;i++){
        for(int j=sizeBlock;j<imageR.cols-sizeBlock;j++){

            costMin = 1000000;
            costCurrent = 0.0;
            disparity = 0;

            /// get right block
            getBlock(&imageR,i,j,blockR,sizeBlock);

            /// determined the maximum disparity for edge the image
            int rangeDisparityCurrent = rangeDisparity;
            if( j - (imageR.cols-sizeBlock - rangeDisparity)>0 )
                rangeDisparityCurrent = (imageR.cols-sizeBlock) - j;

            /// range the disparity
            for(int d=0;d<rangeDisparityCurrent;d++){
                /// get right left
                getBlock(&imageL,i,j+d,blockL,sizeBlock);

                /// calculate cost
                costCurrent = calculateCost(blockL,blockR,lengthBlock);

                /// find minimum cost
                if( costCurrent < costMin ){
                    disparity = d;
                    costMin = costCurrent;
                }
            }
            /// normalizing the disparity
            result.data[i * imageR.cols + j] = disparity*normalize;
        }
    }

    /// free memory
    delete[] blockR;
    delete[] blockL;
}
/**
 * Function that calculated the cost volume (for optimize)
 */
float*** calculateVolumeCost(Mat& imageL,Mat& imageR,Mat& result,int rangeDisparity){

    float ***volumeCost = new float **[rangeDisparity];

    int channel = imageR.channels();

    int numColumsRGB = imageR.cols*3;///image->step;
    for(int d=0;d<rangeDisparity;d++){
        volumeCost[d] = new float*[imageR.rows];
        int displacement = d*channel;
        for(int i=0;i<imageR.rows;i++){
            volumeCost[d][i] = new float[imageR.cols];
            for(int j=0;j<imageR.cols-d;j++){
                int index = i*numColumsRGB + j*channel;

                /// calculate the cost for each channel
                float costR = funCost((float)imageR.data[ index ] - (float)imageL.data[ index + displacement]);
                float costG = funCost((float)imageR.data[ index +1] - (float)imageL.data[ index + 1+displacement]);
                float costB = funCost((float)imageR.data[ index +2] - (float)imageL.data[ index + 2+displacement]);

                /// cost total the pixel
                volumeCost[d][i][j] = (costR + costG + costB)/channel;
            }
        }
    }
    return volumeCost;
}
/**
 * Function that calculated the disparity map, pre-calculated cost (optimize)
 */
void disparityObtimized( Mat& imageL,Mat& imageR,Mat& result,int rangeDisparity,int sizeBlock){

    result = Mat(imageR.size(), CV_8UC1, cv::Scalar(0));

    /// pre calculated the volume cost for disparities
    float ***volumeCost = calculateVolumeCost(imageL,imageR,result,rangeDisparity);

    float costMin = 1000000;
    int disparity = 0;
    float costCurrent = 0.0;

    /// factor of normalization
    float normalize = 255.0f/rangeDisparity;

    for(int i=sizeBlock;i<imageR.rows-sizeBlock;i++){
        int iRow = i * imageR.cols;
        for(int j=sizeBlock;j<imageR.cols-sizeBlock;j++){

            costMin = 1000000;
            costCurrent = 0.0;
            disparity = 0;

            /// determined the maximum disparity for edge the image
            int rangeDisparityCurrent = rangeDisparity;
            if( j - (imageR.cols-sizeBlock - rangeDisparity)>0 )
                rangeDisparityCurrent = (imageR.cols-sizeBlock) - j;

            /// range the disparity
            for(int d=0;d<rangeDisparityCurrent;d++){
                /// calculate cost aggregation
                costCurrent = funAggregation(&imageR,volumeCost[d],i,j,sizeBlock);

                /// find minimum cost
                if( costCurrent < costMin ){
                    disparity = d;
                    costMin = costCurrent;
                }
            }
            /// normalizing the disparity
            result.data[iRow + j] = disparity*normalize;
        }
    }

    /// free memory
    for(int d=0;d<rangeDisparity;d++){
        for(int i=0;i<imageR.rows;i++){
            delete[]volumeCost[d][i];
        }
        delete[]volumeCost[d];
    }
    delete[]volumeCost;
}

/**
 * Structure for interchange of the data (for parallelize)
 */
struct ThreadArgs{
    Mat* imageR;        /// image right
    Mat* result;        /// result disparity map
    float ***volumeCost;/// volume cost with all disparities
    int imin;           /// first row to evaluate
    int imax;           /// last row to evaluate
    int rangeDisparity; /// range max for the disparity
    int sizeBlock;      /// size block
};
/**
 * Function that calculated the disparity at row of the image(for parallelize)
 */
void disparityForLine(void *ptr){

    /// receiving the arguments
    struct ThreadArgs *args = (struct ThreadArgs *)ptr;

    float costMin = 1000000;
    int disparity = 0;
    float costCurrent = 0.0;

    /// factor of normalization
    float normalize = 255.0f/args->rangeDisparity;

    for(int i=args->imin;i<args->imax;i++){
        int iRow = i * args->result->cols;
        for(int j=args->sizeBlock;j<args->result->cols-args->sizeBlock;j++){
            costMin = 1000000;
            costCurrent = 0.0;
            disparity = 0;

            /// determined the maximum disparity for edge the image
            int rangeDisparityCurrent = args->rangeDisparity;

            if( j - (args->result->cols-args->sizeBlock - args->rangeDisparity)>0 )
                rangeDisparityCurrent = (args->result->cols-args->sizeBlock) - j;

            /// range the disparity
            for(int d=0;d<rangeDisparityCurrent;d++){
                /// calculate cost aggregation
                costCurrent = funAggregation(args->imageR,args->volumeCost[d],i,j,args->sizeBlock);

                /// find minimum cost
                if( costCurrent < costMin ){
                    disparity = d;
                    costMin = costCurrent;
                }
            }
            /// normalizing the disparity
            args->result->data[iRow + j] = disparity*normalize;
        }
    }
    return;
}
/**
 * Function that calculated the disparity (optimized parallel)
 */
void disparityParallel( Mat& imageL,Mat& imageR,Mat& result,int rangeDisparity,int sizeBlock){

    result = Mat(imageR.size(), CV_8UC1, cv::Scalar(0));

    /// pre calculated the volume cost for disparities
    float ***volumeCost = calculateVolumeCost(imageL,imageR,result,rangeDisparity);



    int step = 50;
    /// calculating number of threads
    int numThreads = (imageR.rows - 2*sizeBlock )/ step;
    /// number of rows missing consider
    int rest = (imageR.rows - 2*sizeBlock ) % step;


    //printf("number of threads: %d\n",numThreads);

    /// initializing the thread and
    thread *tt = new thread[numThreads];
    struct ThreadArgs *args = new struct ThreadArgs[numThreads] ;

    for(int i = 0,imin = sizeBlock; i < numThreads;i++){
        /// preparing the arguments for a thread
        args[i] = {&imageR,&result,volumeCost,imin,imin+step+rest,rangeDisparity,sizeBlock};
        /// creating a thread
        tt[i] = thread(disparityForLine,&args[i]);
        //pthread_create(&tt[i], NULL, disparityForLine, &args[i]);

        imin+=step+rest;
        rest = 0;
    }

    for(int i = 0; i < numThreads;i++){
        //pthread_join(tt[i], NULL);
        tt[i].join();
    }
    /// free memory
    for(int d=0;d<rangeDisparity;d++){
        for(int i=0;i<imageR.rows;i++){
            delete[]volumeCost[d][i];
        }
        delete[]volumeCost[d];
    }
    delete[]volumeCost;
}


#endif // STEREOVISION_H_INCLUDED
