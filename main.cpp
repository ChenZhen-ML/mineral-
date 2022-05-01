#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

void polygonDetect(Mat &srcImg, double epsilon, int minAcreage, int maxAcreage);
float ratio_wl(RotatedRect rrect);//计算外接矩形的长宽比

int wl[2]={720,1280};//720,1280;1280,720;480,270;

float cal_cos(float x1,float x2,float x3,float x4);//the cost of vector<x1,x2>and<x3,x4>
bool rect_slope(vector<Point2f> Points);//针对正方形
bool remove_parallelogram(vector<Point2f> Points);//排除平行四边形
void draw_contours(RotatedRect rrect);//画出外接矩形
float squareRoot(float& d1,float& d2,float& d3,float& d4);//求4个数的标准差
float average(float& a1,float& a2,float& a3,float& a4);//求4个数的均值
bool target_test(vector<Point2f> Points);//目标正方形边长的标准差是否小于max_squareRoot
int getcross(Point2f a,Point2f b,Point2f p);
bool isin(vector<Point2f> Points,Point2f p);//点p是否在点集所围成的多边形内部
int rectangel(int form[],int p1,int p2,int p3,int p4);
int which_kind(vector<Point2f> contours_center,vector<int> tar_more,vector<Point2f> Points);
void my_rotate(int type,vector<Point2f> Points,vector<Point2f> contours_center,int form[],int p1,int p2,int p3,int p4);
Point2f aver_center(vector<Point2f>P);

float max_squareRoot=100;//最大标准差
float up_down=2;//上下旋转的角度，上为正
float left_right=2;//左右旋转的角度，左为正
float center_rotate=0;//中心旋转的角度，逆时针为正
int type=0;//识别到的面的类型，1,2,3分别代表空白，二维码，R

Mat image,frame,hsv,gray, binary, mask;
int main()
{
    VideoCapture capture(1);
    if (!capture.isOpened())//判断是否读取成功
        std::cout << "please make sure the route" << std::endl;
    capture.set(CAP_PROP_FRAME_HEIGHT,480);
    capture.set(CAP_PROP_FRAME_WIDTH,640);

    //Mat frame;
    Mat cameraMatrix=(Mat_<float>(3,3)<<638.9910,0,320.4412,
            0,639.3573,254.3665,
            0,0,1);
    Mat distCoeffs=(Mat_<float>(1,4)<<-0.5024,0.2813,0,0);
    while (true) {
        capture >> image;            //或者capture.read(frame),将视频的图片一帧一帧传给frame矩阵
        undistort(image,frame,cameraMatrix,distCoeffs);//去畸变

        if (frame.empty())//传至视频结束，矩阵为空，退出
            break;
        //cout<<"length"<<capture.get(CAP_PROP_FRAME_HEIGHT);//1280
        // cout<<"width"<<capture.get(CAP_PROP_FRAME_WIDTH);  //720

        polygonDetect(frame, 9, 0, 100000);

        /*if(left_right==0&&up_down==0) {
            cout<<"stop"<<endl; //break;
        }
        if(left_right==2&&up_down==2) continue;
        else {
            cout<<"left_right<<"<<left_right<<endl;
            cout<<"up_down<<"<<up_down<<endl;
            //break;
        }*/

        imshow("frame",frame);

        int c = waitKey(10);
        if (c == 27)
            break;
    }
        return 0;
}

//多边形的检测
void polygonDetect(Mat &srcImg, double epsilon, int minAcreage, int maxAcreage)
{
    //彩色图转灰度图
    cvtColor(srcImg, gray, COLOR_BGR2GRAY);

    GaussianBlur(gray,gray,Size(5,5),10,20);
    threshold(gray,binary,100,255,THRESH_BINARY);

    Mat kernel= getStructuringElement(0,Size(2,2));
    erode(binary,binary,kernel,Point(-1,-1),3);
    //morphologyEx(binary, binary, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(5, 5)), Point(-1, -1));
    // 开运算
    //morphologyEx(binary, binary, MORPH_OPEN, getStructuringElement(MORPH_RECT, Size(3, 3)), Point(-1, -1));
    //Canny(binary, binary, 3, 9, 3);

    //轮廓发现与绘制
    vector<vector<Point>> contours;//轮廓
    vector<Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point());
    vector<float> contours_realarea;//轮廓面积
    for(int t=0;t<contours.size();t++) {
        drawContours(srcImg, contours, t, Scalar(255, 0, 0), 2, 8);
        float area= contourArea(contours[t]);
        contours_realarea.push_back(area);
    }
    vector<float> contours_ratio_wl;//长宽比
    vector<float> contours_area;//外接矩形面积
    vector<Point2f> contours_center;//外接矩形中心
    int form[contours.size()];
    vector<Point2f> result;

    for(int t=0;t<contours.size();t++)
    {
        RotatedRect rrect= minAreaRect(contours[t]);//最小外接矩形

        draw_contours(rrect);
        float ratio=ratio_wl(rrect);
        contours_ratio_wl.push_back(ratio);//外接矩形的长宽比

        approxPolyDP(contours[t],result,epsilon,true);//求多边形的边数
        Point2f center= aver_center(result);
        //Point2f center=rrect.center;//外接矩形的中心
        contours_center.push_back(center);//将中心点保存
        float area=(rrect.size.height*rrect.size.width);//计算外接矩形的尺寸，以便筛选合适的轮廓
        contours_area.push_back(area);
        //circle(srcImg,center,2,Scalar(0,255,0),2,8,0);

        //approxPolyDP(contours[t],result,epsilon,true);//求多边形的边数
        if(result.size()==6&&contours_realarea[t]>minAcreage&&contours_realarea[t]<maxAcreage)//如果为6条边符合一类
        {
            putText(srcImg,"0",center,0,1,Scalar(0,255,0),1,8);
           // putText(srcImg,ratio,center,0,1,Scalar(0,255,0),1,8);
            form[t]=0;
        }
        else if(result.size()==4&& rect_slope(result)&&contours_realarea[t]>minAcreage&&contours_realarea[t]<maxAcreage)//如果为4条边符合另一类
        {

            putText(srcImg,"1",center,0,1,Scalar(0,255,0),1,8);
            form[t]=1;
        }
        else
        {
            //putText(srcImg,"-1",center,0,1,Scalar(0,0,255),1,8);
            form[t]=-1;
        }
    }

    vector<int> tar_more;//包含中间二维码类型
    vector<int> tar;
    for(int i=0;i<contours.size();i++)
    {
        if(form[i]==0||form[i]==1)
        {
            tar_more.push_back(i);
            if(contours_ratio_wl[i]<1.5)
            tar.push_back(i);//保存需要的类型对应的轮廓序号
        }
    }

    float temp_dist;
    vector<float>dist;
    if(tar.size()>=4)
    {
        for(int i=0;i<tar.size();i++)
        {
            temp_dist=(contours_center[tar[i]].x-wl[0]/2)*(contours_center[tar[i]].x-wl[0]/2)+(contours_center[tar[i]].y-wl[1]/2)*(contours_center[tar[i]].y-wl[1]/2);
            dist.push_back(temp_dist);
        }
        int min1,min2,min3,min4;//距离中心最近的4个点
        temp_dist = 1000000;
        for (int i = 0; i < dist.size(); i++) {
            if (dist[i] < temp_dist) {
                min1=i;
                temp_dist = dist[i];
            }
           }
        circle(binary,contours_center[tar[min1]],2,Scalar(0,0,0),2,8,0);

        temp_dist = 1000000;
        for (int i = 0; i < dist.size(); i++) {
            if (dist[i] < temp_dist&&i!=min1) {
                min2=i;
                temp_dist = dist[i];
            }
        }
        circle(binary,contours_center[tar[min2]],2,Scalar(0,0,0),2,8,0);

        temp_dist = 1000000;
        for (int i = 0; i < dist.size(); i++) {
            if (dist[i] < temp_dist&&i!=min1&&i!=min2) {
                min3=i;
                temp_dist = dist[i];
            }
        }
        circle(binary,contours_center[tar[min3]],2,Scalar(0,0,0),2,8,0);

        temp_dist = 1000000;
        for (int i = 0; i < dist.size(); i++) {
            if (dist[i] < temp_dist&&i!=min1&&i!=min2&&i!=min3) {
                min4=i;
                temp_dist = dist[i];
            }
        }

        vector<Point2f> tar_Points;
        tar_Points.push_back(contours_center[tar[min1]]);
        tar_Points.push_back(contours_center[tar[min2]]);
        tar_Points.push_back(contours_center[tar[min3]]);
        tar_Points.push_back(contours_center[tar[min4]]);
        bool change=rect_slope(tar_Points);

        if(change) {
            if(target_test(tar_Points)&& remove_parallelogram(tar_Points))
            {
                line(binary, contours_center[tar[min1]], contours_center[tar[min2]], 2, 8, 0);
                line(binary, contours_center[tar[min2]], contours_center[tar[min3]], 2, 8, 0);
                line(binary, contours_center[tar[min3]], contours_center[tar[min4]], 2, 8, 0);
                line(binary, contours_center[tar[min4]], contours_center[tar[min1]], 2, 8, 0);
                line(binary, contours_center[tar[min2]], contours_center[tar[min4]], 2, 8, 0);
                line(binary, contours_center[tar[min3]], contours_center[tar[min1]], 2, 8, 0);

                cout<<"1:"<<form[tar[min1]]<<endl;
                cout<<"2:"<<form[tar[min2]]<<endl;
                cout<<"3:"<<form[tar[min3]]<<endl;
                cout<<"4:"<<form[tar[min4]]<<endl;
                if(rectangel(form,tar[min1],tar[min2],tar[min3],tar[min4])==1)
                     type=3;
                else type=which_kind(contours_center,tar_more,tar_Points);}
        }
        else type=4;

        my_rotate(type,tar_Points,contours_center,form,tar[min1],tar[min2],tar[min3],tar[min4]);
        cout<<"up_down<<"<<up_down<<endl;
        cout<<"left_right<<"<<left_right<<endl;
    }
    circle(binary,Point2f(wl[0]/2,wl[1]/2),2,Scalar(0,0,0),2,8,0);
    imshow("binary",binary);

}

//矩形轮廓的长宽比
float ratio_wl(RotatedRect rrect)
{
    if(rrect.size.width>rrect.size.height)
        return rrect.size.width/rrect.size.height;
    else
        return rrect.size.height/rrect.size.width;
}

//the cost of vector<x1,x2>and<x3,x4>
float cal_cos(float x1,float x2,float x3,float x4)
{
    return (x1*x3+x2*x4)/sqrt(pow(x1,2)+pow(x2,2))/sqrt(pow(x3,2)+pow(x4,2));
}

//针对正方形，判断横坐标较小的两个点连线和较大的两个点连线是否平行
bool rect_slope(vector<Point2f> Points)
{
    Point2f temp_Point;

    for(int i=0;i<4;i++)
    {
        for(int j=i+1;j<4;j++)
        {
            if(Points[j].x<Points[i].x)
            {
                temp_Point=Points[i];
                Points[i]=Points[j];
                Points[j]=temp_Point;
            }
        }
    }

    float cos1=cal_cos(Points[0].x-Points[1].x,Points[0].y-Points[1].y,
                       Points[2].x-Points[3].x,Points[2].y-Points[3].y);
    float cos2=cal_cos(Points[0].x-Points[2].x,Points[0].y-Points[2].y,
                       Points[1].x-Points[3].x,Points[1].y-Points[3].y);


    if(abs(abs(cos1)-1)<0.02&&abs(abs(cos2)-1)<0.02)
         return true;
    else return false;
}

bool remove_parallelogram(vector<Point2f> Points)
{
    float dist1=sqrt(pow(Points[0].x-Points[2].x,2)+pow(Points[0].y-Points[2].y,2));
    float dist2=sqrt(pow(Points[1].x-Points[3].x,2)+pow(Points[1].y-Points[3].y,2));
    if(abs(dist1/dist2-1)<0.3) return true;
    else return false;
}
//画出最小外接矩形的轮廓
void draw_contours(RotatedRect rrect)
{
    Point2f points[4];
    rrect.points(points);
    for(int i=0;i<4;i++)
    {
        if(i==3)
        {
            line(frame,points[i],points[0],Scalar(0,0,255),2,8,0);
            break;
        }
        line(frame,points[i],points[i+1],Scalar(0,0,255),2,8,0);
    }
}

//求4个点间的距离
bool target_test(vector<Point2f> Points)
{
    float dis[4];
    dis[0]=sqrt(pow(Points[0].x-Points[1].x,2)+pow(Points[0].y-Points[1].y,2));
    dis[1]=sqrt(pow(Points[0].x-Points[2].x,2)+pow(Points[0].y-Points[2].y,2));
    dis[2]=sqrt(pow(Points[2].x-Points[3].x,2)+pow(Points[2].y-Points[3].y,2));
    dis[3]=sqrt(pow(Points[1].x-Points[3].x,2)+pow(Points[1].y-Points[3].y,2));

    cout<<squareRoot(dis[0],dis[1],dis[2],dis[3])<<endl;
    return true;
    if(squareRoot(dis[0],dis[1],dis[2],dis[3])<max_squareRoot) return true;
    else return false;
}

//求平均距离
float average(float& a1,float& a2,float& a3,float& a4)
{
    float ave;
    ave = (a1+a2+a3+a4)/4;
    return ave;
}

//求四个点距离的标准差
float squareRoot(float& d1,float& d2,float& d3,float& d4)
{
    float b1,b2,b3,b4;
    float sr,aveg;
    aveg = average(b1,b2,b3,b4);
    sr = sqrt((b1-aveg)*(b1-aveg)+(b2-aveg)*(b2-aveg)+(b3-aveg)*(b3-aveg)+(b3-aveg)*(b3-aveg));
    return sr;
}

//判断正方形的个数
int rectangel(int form[],int p1,int p2,int p3,int p4)
{
     return form[p1]+form[p2]+form[p3]+form[p4];
}

//判断图案对应的类型：空白或者二维码
int which_kind(vector<Point2f> contours_center,vector<int> tar_more,vector<Point2f> Points)
{
    int flag1=0,count1=0,flag2=0,count2=0,sum=0;

    for(int i=0;i<contours_center.size();i++)//全部点
    {
        if(isin(Points,contours_center[i]))
        {
            flag1=1,count1++;
        };
    }
    if(count1==0) return 1;

    for(int i=0;i<tar_more.size();i++)//二维码
    {
        if(isin(Points,contours_center[tar_more[i]]))
        {
            flag2=1,count2++;
        };
    }
    if(count2>0) return 2;
    else return 4;//表示错误
}

//判断点是否在Points之内
int getcross(Point2f a,Point2f b,Point2f p){
    return (b.x - a.x) * (p.y - a.y)  - (p.x - a.x) * (b.y - a.y);
}
bool isin(vector<Point2f> Points,Point2f p){
    float boo1=getcross(Points[0],Points[1],p);
    float boo2=getcross(Points[1],Points[3],p);
    float boo3=getcross(Points[3],Points[2],p);
    float boo4=getcross(Points[2],Points[1],p);
    if(boo1>0&&boo2>0&&boo3>0&&boo4>0||boo1<0&&boo2<0&&boo3<0&&boo4<0)
    {
        return true;
    }
    else return false;
}

//翻转情况
void my_rotate(int type,vector<Point2f> Points,vector<Point2f> contours_center,int form[],int p1,int p2,int p3,int p4)
{
    if(type==1) {
        up_down=0;
        left_right=0;
    }
    if(type==2)  {
        up_down=2;
        left_right=0;
    }
    if(type==3) {
        int p=0,flag=0;
        float center_x=(Points[0].x+Points[1].x+Points[2].x+Points[3].x)/4;
        float center_y=(Points[0].y+Points[1].y+Points[2].y+Points[3].y)/4;

        if(form[p1]==1) p=p1;
        if(form[p2]==1) p=p2;
        if(form[p3]==1) p=p3;
        if(form[p4]==1) p=p4;

        if(contours_center[p].y>center_y&&contours_center[p].x>center_x) flag=4;
        if(contours_center[p].y>center_y&&contours_center[p].x<center_x) flag=3;
        if(contours_center[p].y<center_y&&contours_center[p].x>center_x) flag=2;
        if(contours_center[p].y<center_y&&contours_center[p].x<center_x) flag=1;

        /*float tany_x=(contours_center[form[p]].y-center_y)/(contours_center[form[p]].x-center_x);
        float atany_x=atan(tany_x);
        float degree=atany_x*180/3.1415926;*/

        if(flag==4) {up_down=-1;left_right=0;}
        if(flag==3) {up_down=0;left_right=1;}
        if(flag==2) {up_down=0;left_right=-1;}
        if(flag==1) {up_down=1;left_right=0;}
        cout<<"flag<<"<<flag<<endl;
    }
    if(type==4) {
        up_down=2;left_right=2;
    }
}

Point2f aver_center(vector<Point2f>P)
{
    Point2f result;
    for(int i=0;i<P.size();i++)
    {
        result.x+=P[i].x;
        result.y+=P[i].y;
    }
    result.x/=P.size();
    result.y/=P.size();
    return result;
}
































