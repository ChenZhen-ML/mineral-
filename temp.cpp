#include <iostream>
#include <opencv2/opencv.hpp>

#define Up_1 0
#define Left_1 1
#define Right_1 2
#define Down_1 3
#define Down_2 4
#define NoMove 5

using namespace cv;
using namespace std;

int main() {
    Mat demo = imread("../5.png");
    if (demo.empty()) {
        cout << "Image Not found" << endl;
        return -1;
    }

    Mat dst_gray;
    Mat dst_threshold;
    cvtColor(demo, dst_gray, COLOR_BGR2GRAY);
    //threshold(dst_gray, dst_threshold, 110, 255, THRESH_OTSU != 0);
    adaptiveThreshold(dst_gray,dst_threshold,255,ADAPTIVE_THRESH_GAUSSIAN_C,THRESH_OTSU != 0, 201, 13);
    imshow("threshold",dst_threshold);
    waitKey(0);
    Mat pure_threshold = dst_threshold.clone();
    Mat threshold_clone = dst_threshold.clone();
    Mat kernel_mor = getStructuringElement(MORPH_RECT,Size(3,3));
    Mat kernel_dilate = getStructuringElement(MORPH_DILATE,Size(3,3));
    morphologyEx(dst_threshold,dst_threshold,MORPH_OPEN,kernel_mor,Point(-1,-1),3);
    dilate(dst_threshold,dst_threshold,kernel_dilate,Point(-1,-1),4);
    imshow("dilate",dst_threshold);



    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(dst_threshold, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, Point(0, 0));
    drawContours(threshold_clone, contours, -1, Scalar(255, 0, 0), 2, LINE_8);
    vector<RotatedRect> rects;\

    vector<vector<Point>> interested_contours;
    for (auto &contour: contours) {
        rects.push_back(minAreaRect(contour));
        interested_contours.push_back(contour);
    }

    vector<vector<Point>> cornerContours;
    vector<RotatedRect> cornerRect;
    for (int i = 0; i < interested_contours.size(); i++) {
        if (rects[i].size.area() < 20000 && rects[i].size.area() > 3000) {
            if (rects[i].size.height / rects[i].size.width >= 0.8 &&
                rects[i].size.height / rects[i].size.width <= 1.2) {
                Point2f points1[4];
                rects[i].points(points1);
                for (int j = 0; j < 4; j++) {
                    line(demo, points1[j], points1[(j + 1) % 4], Scalar(0, 0, 255));
                }
                cornerRect.push_back(rects[i]);
                cornerContours.push_back(interested_contours[i]);
            }
        }
    }

    //need to correct
    if (cornerContours.size() > 4) {
        vector<vector<Point>> selectMatchContours = cornerContours;
        cornerContours.clear();
        vector<RotatedRect> interested_rects;
        for (auto &i : selectMatchContours) {
            interested_rects.push_back(minAreaRect(i));
        }
        int judge = 0; //from here
        for (int i = 0; i < interested_rects.size(); i++) {
            if (abs(interested_rects[i].angle) <= 5) {
                cornerContours.push_back(selectMatchContours[i]);
            }
        }
    } else if (cornerContours.size() < 4) {
        return -2;
    }


    Point2f point2F[4][4];
    vector<RotatedRect> interestedRect;
    vector<Point> centers;
    vector<int> number;
    int num = 0;
    for (int i = 0; i < 4; i++) {
        cornerRect[i].points(point2F[i]);
        Point center(0, 0);
        for (int n = 0; n < 4; n++) {
            center.x += point2F[i][n].x;
            center.y += point2F[i][n].y;
        }
        center.x /= 4;
        center.y /= 4;
        centers.push_back(center);
        int check = 0;
        double area = contourArea(cornerContours[i]);
        if (cornerRect[i].size.area() / area >= 1 && cornerRect[i].size.area() / area <= 1.5) {
            check++;
            num = i;
        }
        if (check != 0) {
            interestedRect.push_back(cornerRect[i]);
            number.push_back(i);
        }
    }


    for (auto &i: number) {
        circle(demo, centers[i], 2, Scalar(0, 255, 0), 2, LINE_8);
    }


    Point centerofcenter(0, 0);
    for (int i = 0; i < 4; i++) {
        centerofcenter.x += centers[i].x;
        centerofcenter.y += centers[i].y;
    }
    centerofcenter.x /= 4;
    centerofcenter.y /= 4;
    imshow("demo",demo);
    waitKey(0);

    if (interestedRect.size() == 2) {
        Mat kernel = getStructuringElement(MORPH_DILATE, Size(3, 3));
        dilate(dst_threshold, dst_threshold, kernel, Point(-1, -1), 4);

        vector<vector<Point>> contours_barcode;
        findContours(dst_threshold, contours_barcode, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        drawContours(dst_threshold,contours_barcode,-1,Scalar(0,255,0),1,LINE_8);
        imshow("dawijodn", dst_threshold);
        int check = 0;
        for (auto &i: contours_barcode) {
            if (contourArea(i) >= 10000 && contourArea(i) <= 100000) {
                RotatedRect centerRect;
                centerRect = minAreaRect(i);
                if(centerRect.size.area() / contourArea(i) <= 2 && centerRect.size.area() / contourArea(i) >= 1) {
                    if ((centerRect.size.height / centerRect.size.width >= 0.5 &&
                         centerRect.size.height / centerRect.size.width <= 0.73) ||
                        (centerRect.size.height / centerRect.size.width >= 1.5 &&
                         centerRect.size.height / centerRect.size.width <= 1.65)) {
                        check = 1;
                        Point2f point[4];
                        centerRect.points(point);
                        for (int j = 0; j < 4; j++) {
                            line(demo, point[j], point[(j + 1) % 4], Scalar(0, 255, 0), 1, LINE_8);
                        }
                        break;
                    }
                }
            }
        }
        imshow("hahahad", demo);
        waitKey(0);
        if (check == 1) {
            return NoMove;
        } else {
            return Down_2;
        }
    } else {
        imshow("hahahad", demo);
        if (centers[num].x < centerofcenter.x) {
            if (centers[num].y < centerofcenter.y) {
                return Right_1;
            } else {
                return Up_1;
            }
        } else {
            if (centers[num].y < centerofcenter.y) {
                return Down_1;
            } else {
                return Left_1;
            }
        }
    }
}