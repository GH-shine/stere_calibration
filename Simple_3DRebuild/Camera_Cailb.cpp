#include "iostream"
#include <stdio.h>
#include <string.h>
#include <opencv2/opencv.hpp>

#define image_num             13
#define DISPLAY_MONOCULAR     true

using namespace std;
using namespace cv;

char img_filename[100];


static bool readStringList( const string& filename, vector<string>& l )
{
    l.resize(0);
    FileStorage fs(filename, FileStorage::READ);
    if( !fs.isOpened() )
        return false;
    FileNode n = fs.getFirstTopLevelNode();
    if( n.type() != FileNode::SEQ )
        return false;
    FileNodeIterator it = n.begin(), it_end = n.end();
    for( ; it != it_end; ++it )
        l.push_back((string)*it);
    return true;
}

static void StereoCalib(const vector<string>& imagelist, Size boardSize, bool useCalibrated=true, bool showRectified=true)
{
    if( imagelist.size() % 2 != 0 )
    {
        cout << "Error: the image list contains odd (non-even) number of elements\n";
        return;
    }

    bool displayCorners = true;   //可视化角点
    const int maxScale = 2;      
    const float squareSize = 20.f;  // Set this to your actual square size  正方形大小
    // ARRAY AND VECTOR STORAGE:

    vector<vector<Point2f> > imagePoints[2];  //角点像素坐标
    vector<vector<Point3f> > objectPoints;    // 物理坐标
    Size imageSize;

    int i, j, k, nimages = (int)imagelist.size()/2;  //单侧图像的数量

    imagePoints[0].resize(nimages);  //根据nimages确定vector的大小
    imagePoints[1].resize(nimages);  //同上
    vector<string> goodImageList;    

    
    for( i = j = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            const string  filename = imagelist[i*2+k];
//  	    /*char img_path[100] = "/home/shine/slam_ws/slam_pro/data/";
// 	    strc*/at(img_path, filename);
	//  sprintf(image_filename, "/home/shine/slam_ws/slam_pro/data/%p", &filename);
	    
            Mat img = imread(filename, 0);
            if(img.empty())
                break;
            if(imageSize == Size())
                imageSize = img.size();
            else if( img.size() != imageSize )  //确定所有图像大小一样
            {
                cout << "The image " << filename << " has the size different from the first image size. Skipping the pair\n";
                break;
            }   
            bool found = false;
            vector<Point2f>& corners = imagePoints[k][j];
            for( int scale = 1; scale <= maxScale; scale++ )
            {
                Mat timg;
                if( scale == 1 )
                    timg = img;
                else
                    resize(img, timg, Size(), scale, scale);
		
                found = findChessboardCorners(timg, boardSize, corners,
                    CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);		
                if( found )
                {
                    if( scale > 1 )
                    {
                        Mat cornersMat(corners);
                        cornersMat *= 1./scale;
                    }
                    break;
                }
            }
            if( displayCorners )
            {
                cout << filename << endl;
                Mat cimg, cimg1;
                cvtColor(img, cimg, COLOR_GRAY2BGR);
                drawChessboardCorners(cimg, boardSize, corners, found);  //显示角点
                double sf = 640./MAX(img.rows, img.cols);
                resize(cimg, cimg1, Size(), sf, sf);   //调整图像大小640 x 640
                imshow("corners", cimg1);
                char c = (char)waitKey(500);
                if( c == 27 || c == 'q' || c == 'Q' ) //Allow ESC to quit
                    exit(-1);
            }
            else
                putchar('.');
            if( !found )  // whether scale = 1 or 2  can not calculate conners.
                break;
            cornerSubPix(img, corners, Size(11,11), Size(-1,-1),
                         TermCriteria(TermCriteria::COUNT+TermCriteria::EPS,
                                      30, 0.01));
        }
        if( k == 2 )  //
        {
            goodImageList.push_back(imagelist[i*2]);  // Add data to the end of the %vector
            goodImageList.push_back(imagelist[i*2+1]);
            j++;
        }
    }
    cout << j << " pairs have been successfully detected.\n";
    nimages = j;
    if( nimages < 2 )
    {
        cout << "Error: too little pairs to run the calibration\n";
        return;
    }

    imagePoints[0].resize(nimages);    //根据nimages确定vector的大小
    imagePoints[1].resize(nimages);    //同上
    objectPoints.resize(nimages);      //同上

    for( i = 0; i < nimages; i++ )
    {
        for( j = 0; j < boardSize.height; j++ )
            for( k = 0; k < boardSize.width; k++ )
                objectPoints[i].push_back(Point3f(j*squareSize, k*squareSize, 0));
    }

    cout << "Running stereo calibration ...\n";

    Mat cameraMatrix[2], distCoeffs[2];
//*************************************************//
//     FileStorage  fs1;
//     fs1.open("../mydata/intrinsics_Left.xml", FileStorage::READ);
//     
//     Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
//     fs1["camera_matrix"] >> intrinsic_matrix_loaded;
//     fs1["distortion_cofficients"] >> distortion_coeffs_loaded;
//     fs1.release();
//     
//     cameraMatrix[0] = intrinsic_matrix_loaded;
//     distCoeffs[0] = distortion_coeffs_loaded;
//    
//     fs1.open("../mydata/intrinsics_Right.xml", FileStorage::READ);
//     fs1["camera_matrix"] >> intrinsic_matrix_loaded;
//     fs1["distortion_cofficients"] >> distortion_coeffs_loaded;
//     fs1.release(); 
//    
//     cameraMatrix[1] = intrinsic_matrix_loaded; 
//     distCoeffs[1] = distortion_coeffs_loaded;
    
//******************************************************//  
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);   // 未先进行单目标定？？？  
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;

    double rms = stereoCalibrate(objectPoints, imagePoints[0], imagePoints[1],
                    cameraMatrix[0], distCoeffs[0],
                    cameraMatrix[1], distCoeffs[1],
                    imageSize, R, T, E, F,
                    CALIB_FIX_ASPECT_RATIO +
                    CALIB_ZERO_TANGENT_DIST +
                    CALIB_SAME_FOCAL_LENGTH +
                    CALIB_RATIONAL_MODEL +
                    CALIB_FIX_K3 + CALIB_FIX_K4 + CALIB_FIX_K5,
                    TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 100, 1e-5) );
    cout << "done with RMS error=" << rms << endl;

// CALIBRATION QUALITY CHECK
// because the output fundamental matrix implicitly 
// includes all the output information,
// we can check the quality of calibration using the
// epipolar geometry constraint: m2^t*F*m1=0  极线约束检验
    double err = 0;
    int npoints = 0;
    vector<Vec3f> lines[2];
    for( i = 0; i < nimages; i++ )
    {
        int npt = (int)imagePoints[0][i].size();
        Mat imgpt[2];
        for( k = 0; k < 2; k++ )
        {
            imgpt[k] = Mat(imagePoints[k][i]);
            undistortPoints(imgpt[k], imgpt[k], cameraMatrix[k], distCoeffs[k], Mat(), cameraMatrix[k]);
            computeCorrespondEpilines(imgpt[k], k+1, F, lines[k]);
        }
        for( j = 0; j < npt; j++ )
        {
            double errij = fabs(imagePoints[0][i][j].x*lines[1][j][0] +
                                imagePoints[0][i][j].y*lines[1][j][1] + lines[1][j][2]) +
                           fabs(imagePoints[1][i][j].x*lines[0][j][0] +
                                imagePoints[1][i][j].y*lines[0][j][1] + lines[0][j][2]);
            err += errij;
        }
        npoints += npt;
    }
    cout << "average reprojection err = " <<  err/npoints << endl;

    // save intrinsic parameters
    FileStorage fs("../mydata/intrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "M1" << cameraMatrix[0] << "D1" << distCoeffs[0] <<
            "M2" << cameraMatrix[1] << "D2" << distCoeffs[1];
        fs.release();
    }
    else
        cout << "Error: can not save the intrinsic parameters\n";

    Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];

    stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 1, imageSize, &validRoi[0], &validRoi[1]);

    fs.open("../mydata/extrinsics.yml", FileStorage::WRITE);
    if( fs.isOpened() )
    {
        fs << "R" << R << "T" << T << "R1" << R1 << "R2" << R2 << "P1" << P1 << "P2" << P2 << "Q" << Q;
        fs.release();
    }
    else
        cout << "Error: can not save the extrinsic parameters\n";

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

// COMPUTE AND DISPLAY RECTIFICATION
    if( !showRectified )
        return;

    Mat rmap[2][2];
// IF BY CALIBRATED (BOUGUET'S METHOD)
    if( useCalibrated )
    {
        // we already computed everything
    }
// OR ELSE HARTLEY'S METHOD
    else
 // use intrinsic parameters of each camera, but
 // compute the rectification transformation directly
 // from the fundamental matrix
    {
        vector<Point2f> allimgpt[2];
        for( k = 0; k < 2; k++ )
        {
            for( i = 0; i < nimages; i++ )
                std::copy(imagePoints[k][i].begin(), imagePoints[k][i].end(), back_inserter(allimgpt[k]));
        }
        F = findFundamentalMat(Mat(allimgpt[0]), Mat(allimgpt[1]), FM_8POINT, 0, 0);
        Mat H1, H2;
        stereoRectifyUncalibrated(Mat(allimgpt[0]), Mat(allimgpt[1]), F, imageSize, H1, H2, 3);

        R1 = cameraMatrix[0].inv()*H1*cameraMatrix[0];
        R2 = cameraMatrix[1].inv()*H2*cameraMatrix[1];
        P1 = cameraMatrix[0];
        P2 = cameraMatrix[1];
    }

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;
    double sf;
    int w, h;
    if( !isVerticalStereo )
    {
        sf = 600./MAX(imageSize.width, imageSize.height);  //600
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else
    {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }

    for( i = 0; i < nimages; i++ )
    {
        for( k = 0; k < 2; k++ )
        {
            Mat img = imread(goodImageList[i*2+k], 0), rimg, cimg;
            remap(img, rimg, rmap[k][0], rmap[k][1], INTER_LINEAR);
	    
// 	    if(k==0) imshow("Left image rectity", rimg);
// 	    else imshow("Right image rectity", rimg);
	    
            cvtColor(rimg, cimg, COLOR_GRAY2BGR);
            Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
            resize(cimg, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);
            if( useCalibrated )
            {
                Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
                          cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
                rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
            }
        }

        if( !isVerticalStereo )
            for( j = 0; j < canvas.rows; j += 16 )
                line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
        else
            for( j = 0; j < canvas.cols; j += 16 )
                line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
        imshow("rectified", canvas);
	imwrite("../mydata/rectified.jpg", canvas);
        char c = (char)waitKey();
        if( c == 27 || c == 'q' || c == 'Q' )
            break;
    }
}



void Stereo_cailb(void)
{
    Size boardSize = Size(9, 6);  //角点的个数
    string imagelistfn = "../mydata/stereo_calib_opencv.xml";  //待处理图像的路径信息
    bool showRectified = true;  //进行矫正

    vector<string> imagelist;
    bool ok = readStringList(imagelistfn, imagelist);
    if(!ok || imagelist.empty())
    {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
    }
    
    StereoCalib(imagelist, boardSize, true, showRectified);
}





void monocular_cailb_L(void)
{
  float image_sf = 0.5;
  int board_w = 9;  //  每行角点的个数
  int board_h = 6;  // 没列角点的个数
  int board_n = board_w*board_h;  //每幅图角点的总个数
  int n_boards;  
  Size board_sz = Size(board_w, board_h);
  Size image_size; // 图像尺寸
    
  vector< vector<Point2f> > image_points;  //所有待标定图像像素坐标
  vector< vector<Point3f> > object_points;  //所有待标定图像像物理坐标
  
  string imagelistfn = "../mydata/stereo_calib_opencv.xml";  //待处理图像的路径信息

  vector<string> imagelist;
  bool ok = readStringList(imagelistfn, imagelist);
  if(!ok || imagelist.empty())
  {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
  }
  int nimages = (int)imagelist.size();  //图像的数量
  
  n_boards = nimages/2;
  
  for(int i = 0; i < nimages; i=i+2)
  {
  //  sprintf(img_filename, "/home/shine/slam_ws/slam_pro/Simple_3DRebuild/data/left%d.jpg", i);
    const string  img_filename = imagelist[i];
    
    Mat image0 = imread(img_filename);
    Mat image;
    image_size = image0.size();
    resize(image0, image, Size(), image_sf, image_sf, CV_INTER_LINEAR);  //重新调整图像的大小
    
    vector<Point2f> corners;  //每幅图像角点的位置信息
    bool found = findChessboardCorners(image, board_sz, corners);  //角点检测 
    
    drawChessboardCorners(image, board_sz, corners, found);  //画出角点位置

    if(found)
    {
      image ^=Scalar::all(255); // 
      Mat mcorners(corners);
      mcorners *= (1./image_sf); //
      image_points.push_back(corners); //
      object_points.push_back(vector<Point3f>());
      vector<Point3f>& opts = object_points.back();
      opts.resize(board_n);
      for(int j = 0; j<board_n; j++)
      {
          opts[j] = Point3f((float)(j/board_w), (float)(j%board_w), 0.f);
      }
       cout << "Collected our" << (int)image_points.size() << "of" << n_boards << "needed chessboar\n" <<endl;
     }
     
     if(DISPLAY_MONOCULAR)  //可视化角点
     {
	imshow("Cailbration", image);
	imwrite("../mydata/Cailbration_L.jpg",image);
	waitKey(600);
     }

   }
   
   cout << "\n\n***Calibrating the camera...\n\n" <<  endl;
   
   Mat intrinsic_matrix, distortion_coeffs;
   
   double err = calibrateCamera(
     object_points,
     image_points,
     image_size,
     intrinsic_matrix,
     distortion_coeffs,
     noArray(),
     noArray(),
     CALIB_ZERO_TANGENT_DIST | CALIB_FIX_PRINCIPAL_POINT
   );
   
   cout << "***DONE! \n\nReprojection error is" << err <<
        "\n Storing Intrinsics_Left.xml file \n\n";
   
   FileStorage fs("../mydata/Intrinsics_Left.xml", FileStorage::WRITE);
   
   fs << "image_width" << image_size.width << "image_height" <<
      image_size.height << "camera_matrix" << intrinsic_matrix << 
      "distortion_cofficients" << distortion_coeffs;
   fs.release();
   
   fs.open("../mydata/Intrinsics_Left.xml", FileStorage::READ);
   Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
   fs["camera_matrix"] >> intrinsic_matrix_loaded;
   fs["distortion_cofficients"] >> distortion_coeffs_loaded;
   
   cout <<"\nintrinsic matrix: " << intrinsic_matrix_loaded;
   cout << "\ndistortion coefficients: "<< distortion_coeffs_loaded;
   
   
   cout << "\n\n***Rectify Images...\n\n" <<  endl;
   
   Mat map1, map2;
   initUndistortRectifyMap(
     intrinsic_matrix_loaded,
     distortion_coeffs_loaded,
     Mat(),
     intrinsic_matrix_loaded,
     image_size,
     CV_16SC2,
     map1,
     map2
   );
   for(int i=0; i < nimages; i=i+2)
   {
     Mat image;
    // sprintf(img_filename, "/home/shine/slam_ws/slam_pro/Simple_3DRebuild/data/left%d.jpg", i);
     
     const string  img_filename = imagelist[i];
     Mat image0 = imread(img_filename);
     remap(
       image0,
       image,
       map1,
       map2,
       INTER_LINEAR,
       BORDER_CONSTANT,
       Scalar()
    );
    if(DISPLAY_MONOCULAR)
    {
	imshow("Undistortd", image);
	imwrite("../mydata/Undistortd_L.jpg",image);
	waitKey(600);
    }
   }
  
}

void monocular_cailb_R(void)
{
  float image_sf = 0.5;
  int board_w = 9;//7;  //  每行角点的个数
  int board_h = 6;//5;  // 没列角点的个数
  int board_n = board_w*board_h;  //每幅图角点的总个数
  int n_boards;  
  Size board_sz = Size(board_w, board_h);
  Size image_size; 
  
  vector< vector<Point2f> > image_points;  //所有待标定图像像素坐标
  vector< vector<Point3f> > object_points;  //所有待标定图像像物理坐标
  string imagelistfn = "../mydata/stereo_calib_opencv.xml";  //待处理图像的路径信息

  vector<string> imagelist;
  bool ok = readStringList(imagelistfn, imagelist);
  if(!ok || imagelist.empty())
  {
        cout << "can not open " << imagelistfn << " or the string list is empty" << endl;
  }
  int nimages = (int)imagelist.size();  //图像的数量
  
  n_boards = nimages/2;
  
  for(int i = 1; i < nimages; i=i+2)
  {
  //    sprintf(img_filename, "/home/shine/slam_ws/slam_pro/Simple_3DRebuild/data/right%d.jpg", i);
    const string  img_filename = imagelist[i];
  
    Mat image0 = imread(img_filename);
    Mat image;
    image_size = image0.size();
    resize(image0, image, Size(), image_sf, image_sf, CV_INTER_LINEAR);  //重新调整图像的大小
    
    vector<Point2f> corners;  //每幅图像角点的位置信息
    bool found = findChessboardCorners(image, board_sz, corners);  //角点检测 
    
    drawChessboardCorners(image, board_sz, corners, found);  //画出角点位置
    
    if(found)
    {
      image ^=Scalar::all(255); // 
      Mat mcorners(corners);
      mcorners *= (1./image_sf); //
      image_points.push_back(corners); //
      object_points.push_back(vector<Point3f>());
      vector<Point3f>& opts = object_points.back();
      opts.resize(board_n);
      for(int j = 0; j<board_n; j++)
      {
          opts[j] = Point3f((float)(j/board_w), (float)(j%board_w), 0.f);
      }
       cout << "Collected our" << (int)image_points.size() << "of" << n_boards << "needed chessboar\n" <<endl;
     }
     if(DISPLAY_MONOCULAR)
     {
	imshow("Cailbration", image);
	imwrite("../mydata/Cailbration_R.jpg",image);
	waitKey(600);
     }
   }
   
   cout << "\n\n***Calibrating the camera...\n\n" <<  endl;
   
   Mat intrinsic_matrix, distortion_coeffs;
   
   double err = calibrateCamera(
     object_points,
     image_points,
     image_size,
     intrinsic_matrix,
     distortion_coeffs,
     noArray(),
     noArray(),
     CALIB_ZERO_TANGENT_DIST | CALIB_FIX_PRINCIPAL_POINT
   );
   
   cout << "***DONE! \n\nReprojection error is" << err <<
        "\n Storing intrinsics_Right.xml file \n\n";
   
   FileStorage fs("../mydata/intrinsics_Right.xml", FileStorage::WRITE);
   
   fs << "image_width" << image_size.width << "image_height" <<
      image_size.height << "camera_matrix" << intrinsic_matrix << 
      "distortion_cofficients" << distortion_coeffs;
   fs.release();
   
   fs.open("../mydata/intrinsics_Right.xml", FileStorage::READ);
   Mat intrinsic_matrix_loaded, distortion_coeffs_loaded;
   fs["camera_matrix"] >> intrinsic_matrix_loaded;
   fs["distortion_cofficients"] >> distortion_coeffs_loaded;
   
   cout <<"\nintrinsic matrix: " << intrinsic_matrix_loaded;
   cout << "\ndistortion coefficients: "<< distortion_coeffs_loaded;
   
   cout << "\n\n***Rectify Images...\n\n" <<  endl;
   
   Mat map1, map2;
   initUndistortRectifyMap(
     intrinsic_matrix_loaded,
     distortion_coeffs_loaded,
     Mat(),
     intrinsic_matrix_loaded,
     image_size,
     CV_16SC2,
     map1,
     map2
   );
   for(int i=1; i < nimages; i=i+2)
   {
     Mat image;
    // sprintf(img_filename, "/home/shine/slam_ws/slam_pro/Simple_3DRebuild/data/right%d.jpg", i);
     const string  img_filename = imagelist[i];
     Mat image0 = imread(img_filename);
     remap(
       image0,
       image,
       map1,
       map2,
       INTER_LINEAR,
       BORDER_CONSTANT,
       Scalar()
    );
    if(DISPLAY_MONOCULAR)
    {
      imshow("Undistortd", image);
      imwrite("../mydata/Undistortd_R.jpg",image);
      waitKey(600);
    }
   }
  
}