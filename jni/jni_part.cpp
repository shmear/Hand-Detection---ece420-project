#include <jni.h>
#include <opencv2/core/core.hpp>
#include <android/log.h>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string.h>



using namespace std;
using namespace cv;
using cv::CLAHE;

extern "C" {

//Global Variables
int erosion_elem = 0;
int erosion_size = 0;
int dilation_elem = 0;
int dilation_size = 0;
double finger_area = 0.0;
int hmin = 0;
int hmax = 25;
int smin = 48;
int smax = 255;

JNIEXPORT int JNICALL Java_org_ece420_lab5_Sample4View_HandSegment(JNIEnv*, jobject, jlong addrRgba, jlong addrHandSegment, jint view_mode)
{
	Mat* pRGB = (Mat*)addrRgba; //source mat
	Mat* pHandSeg = (Mat*)addrHandSegment; //destination mat
	Mat temp_gray; //intermediate matrix
	Mat test;
	vector<Mat> temp_hsv;
	Mat hsv;
	int view = view_mode;
	int row = (*pRGB).rows;
	int columns = (*pRGB).cols;
	int size = row*columns;
	int rgb;
	float R, G, B, r, g, b;
	double clipLimit;
	Size tileGridSize;


	cvtColor(*pRGB,hsv,COLOR_RGB2HSV);

	inRange(hsv,Scalar(hmin, smin, 0),Scalar(hmax, smax, 255),*pHandSeg); // orig h = 20
	split(hsv,temp_hsv);

	Mat element = getStructuringElement( MORPH_CROSS, Size( 3, 3 ), Point( -1, -1 ));
	Mat element2 = getStructuringElement( MORPH_ELLIPSE, Size( 5, 5 ), Point( -1, -1 ));

	erode(*pHandSeg, test, element);
	dilate(test,*pHandSeg,element2);
	medianBlur(*pHandSeg,*pHandSeg,5);
	if (view == 0){
			return 0;
		}


	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	int largest_area=0;
	int largest_contour_index=0;
	double area;

	findContours(*pHandSeg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_NONE, Point(0, 0));

	vector< vector<int> >hull( contours.size() );
	vector< vector<Vec4i> > convexityDefectsSet( contours.size() );
	vector< vector<Point> >hull_points( contours.size() );
	vector< vector<Point> >fingertips( contours.size() );
	vector< vector<Point> >knuckle( contours.size() );
	vector< vector<Point> >convDefPT( contours.size() );
	vector< vector<int>  >filterhullIdx(contours.size());
	vector< vector<int>  >defectIdx(contours.size());

	for (int i = 0; i < contours.size(); i++){
	    area = contourArea(contours[i], false);  //Find the area of contour

	    if(area > largest_area){
	    	largest_area = area;
	    	largest_contour_index = i; //Store the index of largest contour
	    }
	}

	if (largest_area < 16500){
		finger_area = 0;
		hmin = 0;
		hmax = 25;
		smin = 48;
		smax = 255;
	    return 0;
	}

	convexHull( Mat(contours[largest_contour_index]), hull[0], true, false);

	 Scalar color(0,255,0,255);
	 Scalar color2(255,0,0,255);
	 Scalar color3(0,0,255,255);
	 double dist;
	 if (view == 2){
		 drawContours(*pRGB, contours, largest_contour_index, color, 3, 8, vector<Vec4i>(), 0, Point());
	 }
	// approxPolyDP(Mat(contours[largest_contour_index]), contours[largest_contour_index], 9, true);

	 int k = 0, extra_pt = 0;
	 Point pt1, pt2;

	for (int i = 0; i < hull[0].size(); i++){
		int ind = hull[0][i];
	    extra_pt = 0;
	    if (i == 0){
	       filterhullIdx[0].push_back(ind);
	       hull_points[0].push_back(contours[largest_contour_index][ind]);
	       k = 1;
	     }

	    else{
	        for (int j = 0;j<k;j++){

	        int ind2 = filterhullIdx[0][j];
	        pt1 = contours[largest_contour_index][ind];
	        pt2 = contours[largest_contour_index][ind2];
	        dist = sqrt(pow((pt1.x-pt2.x),2) + pow((pt2.y-pt1.y),2));
	        if (dist <= 80){
	        	extra_pt = 1;
	        }

	    }
	       if (extra_pt == 0){
	        filterhullIdx[0].push_back(ind);
	        hull_points[0].push_back(contours[largest_contour_index][ind]);
	        k++;
	    }
	   }
	 }
		drawContours( *pRGB, hull_points, -1, color2, 6, 6, vector<Vec4i>(), 0, Point());
		Rect boundrect;
	    boundrect = boundingRect(Mat(hull_points[0]));
	    rectangle( *pRGB, boundrect.tl(), boundrect.br(), color, 2, 8, 0 );
	    if (view == 2){ // draws circles on all hull points
	    	for (int i = 0; i < hull_points[0].size(); i++){
	    		//circle(*pRGB, hull_points[0][i], 7, color3, -1, 8, 0);
	    	   circle(*pRGB, hull_points[0][i], 7, color3, -1, 8, 0);
	    	}
	    }
	    Scalar color4(255,0,0,255);

	    convexityDefects(Mat(contours[largest_contour_index]), filterhullIdx[0], convexityDefectsSet[0]);
	     int newpoint;
	     k = 0;
	     extra_pt = 0;
	     for (int i = 0; i < convexityDefectsSet[0].size(); i++){
	    	 extra_pt = 0;
	    	 newpoint = convexityDefectsSet[0][i][2];
	    	 pt1 = contours[largest_contour_index][newpoint];
	    	 for (int j = 0; j<filterhullIdx[0].size();j++)
	   	   	   	   {
	    		 	 int Idx = filterhullIdx[0][j];
	    		 	 pt2 = contours[largest_contour_index][Idx];
	    		 	 dist = sqrt(pow((pt1.x-pt2.x),2) + pow((pt2.y-pt1.y),2));
	    		 	if (dist <= 50){
	    		 		extra_pt = 1;
	    		 	}

	   	   	   }
// Add here for array of points of convexity defects for ellipse matching
	    	 if (extra_pt == 0){
	    		 if (view == 2){
	    			 circle(*pRGB, pt1, 7, color4, -1, 8, 0);
	    		 }
	    		 defectIdx[0].push_back(newpoint);
	    		 convDefPT[0].push_back(pt1);
	    		 k++;
	   	   }
	      }
	     if (view == 2)
	    	 if (k >= 5){
	    		 RotatedRect palm;
	    		 palm = fitEllipse(convDefPT[0]);
	    		 circle(*pRGB,palm.center,7,color,-1,8,0);
	    	 }
	     double largest;
	     if (boundrect.height>boundrect.width){
	    	 largest = boundrect.height;
	     }
	     else{
	    	 largest = boundrect.width;
	     }

	     // Find FingerTips
	     int currIdx,smallestIdx,smallestIdx2,diffIdx,smallestdist,smallestdist2,largestdist,largestIdx;
	     uint8_t numb_fingers;
	     Point currpt,normClose,normNext,ptClose,ptNext,testpt,knuckleavg;
	     double dotproduct,angle,lNext,lClose;
	     color = (255,0,0,255);
	     color3 = (0,0,255,255);
	     color2 = (0,0,0,255);
	     numb_fingers = 0;
	     for (int i = 0; i < filterhullIdx[0].size();i++){
	    	 int testIdx = filterhullIdx[0][i];
	    	 testpt = contours[largest_contour_index][testIdx];
	    	 smallestdist = 10000000;
	    	 smallestdist2 = 10000000;
	    	 largestdist = 0;
	    	 smallestIdx = 0;
	    	 smallestIdx2 = 0;
	    	 largestIdx = 0;
	    	 for (int j = 0; j<defectIdx[0].size();j++){
	    		 currIdx = defectIdx[0][j];
	    		 diffIdx = abs(testIdx - currIdx);
	    		 if (diffIdx < smallestdist){
	    			 smallestIdx2 = smallestIdx;
	    			 smallestdist2 = smallestdist;
	    			 smallestIdx = currIdx;
	    			 smallestdist = diffIdx;
	    		 }
	    		 else if (diffIdx < smallestdist2 && diffIdx > smallestdist){
	    			 smallestIdx2 = currIdx;
	    			 smallestdist2 = diffIdx;
	    		 }
	    		 else if (diffIdx > largestdist){
	    			 largestIdx = currIdx;
	    			 largestdist = diffIdx;
	    		 }


	    	 }
	    	 //ptClose = contours[largest_contour_index][smallestIdx];
	    	 if ((smallestIdx > testIdx && testIdx > smallestIdx2) || (smallestIdx < testIdx && testIdx < smallestIdx2)){
	    		 ptNext = contours[largest_contour_index][smallestIdx2];
	    	 }
	    	 else{
	    		 ptNext = contours[largest_contour_index][largestIdx];
	    	 }
	    	 ptClose = contours[largest_contour_index][smallestIdx];
	    	 normClose.x = ptClose.x - testpt.x;
	         normClose.y = ptClose.y - testpt.y;
	    	 normNext.x = ptNext.x - testpt.x;
	    	 normNext.y = ptNext.y - testpt.y;
	    	 dotproduct = normClose.x * normNext.x + normClose.y * normNext.y;
	    	 lClose = sqrt(pow(normClose.x,2)+pow(normClose.y,2));
	    	 lNext = sqrt(pow(normNext.x,2)+pow(normNext.y,2));
	    	 angle = acos(dotproduct/lClose/lNext);
	    	 if (view == 2){
	    		 line(*pRGB,testpt,ptClose,color2,2,8,0);
	    		 line(*pRGB,testpt,ptNext,color3,2,8,0);
	    	 }
	    	 if(smallestIdx != smallestIdx2)
	    		 if (angle < 0.6981) {// 45 degrees
	    			 if ((lClose > 0.28 * largest && lNext > 0.28 * largest) && lClose < 0.65*largest && lNext < 0.65*largest){
						 numb_fingers++;
						 fingertips[0].push_back(testpt);
						 circle(*pRGB, testpt, 20, color, 2, 8, 0);
						 knuckle[0].push_back(knuckleavg);
						 if (view == 1){
						 	   line(*pRGB,testpt,ptClose,color,2,8,0);
						 	   color4 = (0,0,255,255);
						 	   circle(*pRGB, testpt, 7, color4, -1, 8, 0);
						}
				 }
	    			 else if (largest_area <= 0.7*finger_area)
	    				 if (lClose > 0.28 * largest && lNext > 0.28 * largest){
	    					 numb_fingers++;
	    					 fingertips[0].push_back(testpt);
	    					 circle(*pRGB, testpt, 20, color, 2, 8, 0);
	    					 knuckle[0].push_back(knuckleavg);
	    					 if (view == 1){
	    						 line(*pRGB,testpt,ptClose,color,2,8,0);
	    						 color4 = (0,0,255,255);
	    						 circle(*pRGB, testpt, 7, color4, -1, 8, 0);
	    					 }
					 }
			 }

	     }
	     double arearat,areabound;
	     areabound = (boundrect.width)*(boundrect.height);
	     arearat = areabound/finger_area;

	     if (numb_fingers == 5){
	    	 finger_area = areabound;
	    	/* hmin_d = 255; feed back loop attempt, not enough computation...
	    	 hmax_d = 0;
	    	 smin_d = 255;
	    	 smax_d = 0;
	    	 for (int i = (int)leftcorner.x ; i < (int)(leftcorner.x+boundrect.width);i++ ){
	    		 for (int j = (int)leftcorner.y; j < (int)(leftcorner.y+boundrect.height);j++){
	    			 testpt.x = (float)(leftcorner.x + i);
	    			 testpt.y = (float)(leftcorner.y + j);
	    			 if (pointPolygonTest(contours[largest_contour_index],testptfeedback,false)>0){
	    				 if (temp_hsv[0].at<uint8_t>(i,j) > hmax_d)
	    					 hmax_d = temp_hsv[0].at<uint8_t>(i,j);
	    				 else if (temp_hsv[0].at<uint8_t>(i,j) < hmin_d)
	    					 hmin_d = temp_hsv[0].at<uint8_t>(i,j);
	    				 if (temp_hsv[1].at<uint8_t>(i,j) > smax_d)
	    					 smax_d = temp_hsv[1].at<uint8_t>(i,j);
	    				 else if (temp_hsv[1].at<uint8_t>(i,j) < smin_d)
	    					 smin_d = temp_hsv[1].at<uint8_t>(i,j);
	    			 }
	    		 }
	    	}
	    	hmin = hmin*0.75 + (hmin_d)*0.25;
	    	hmax = hmax*0.75 + (hmax_d)*0.25;
	    	smin = smin*0.75 + (smin_d)*0.25;
	    	smax = smax*0.75 + (smax_d)*0.25;*/
	     }
	     //else if (numb_fingers == 0 && largest_area <= 0.7*finger_area){
	     else if (arearat <= 0.40){
	    	 numb_fingers = 6;

	     }
	     else if (numb_fingers == 6){
	    	 numb_fingers == 5;
	     }
	     color = (0,0,0,255);
	     return numb_fingers;

}

}

