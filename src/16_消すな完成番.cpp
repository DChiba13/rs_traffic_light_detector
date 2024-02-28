#include <opencv2/opencv.hpp>
#include <iostream>
#include <experimental/filesystem>
#include <ryusei/common/logger.hpp>
#include <ryusei/common/defs.hpp>
#include <ryusei/common/math.hpp>
#include <fstream>


using namespace project_ryusei;
using namespace cv;
using namespace std;
namespace pr = project_ryusei;
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
namespace fs = std::experimental::filesystem;

// 点群をカメラ画像の座標に合わせる(回転させる)関数のパラメータ
#define X_DIFF 0.1
#define Y_DIFF 0.0
#define Z_DIFF -0.085
#define ROLL 0.0
#define PITCH 0.8
#define YAW -2.5

#define DEG_TO_RAD (M_PI / 180.0)
/* 青Mercury 3D-lidar : pandar_40 */
// const double LIDAR_VIEW_ANGLE_H = 180.0 * DEG_TO_RAD; // 障害物を探索する水平画角
// const double LIDAR_VIEW_ANGLE_V = 23.0 * DEG_TO_RAD; // 障害物を探索する垂直画角
// const double LIDAR_RESOLUTION_H = 0.4 * DEG_TO_RAD; // LiDARの水平解像度(分解能)
// const double LIDAR_RESOLUTION_V = 1.0 * DEG_TO_RAD; // LiDARの垂直解像度(分解能)
/* 赤Mercury  3D-lidar : pandar_xt_32 */
constexpr double LIDAR_VIEW_ANGLE_H = 140.0 * DEG_TO_RAD; // 障害物を探索する水平画角 /* 140 ~ 360° */
constexpr double LIDAR_VIEW_ANGLE_V = 32.0 * DEG_TO_RAD; // 障害物を探索する垂直画角
constexpr double LIDAR_RESOLUTION_H = 0.36 * DEG_TO_RAD; // LiDARの水平解像度(分解能)
constexpr double LIDAR_RESOLUTION_V = 1.0 * DEG_TO_RAD; // LiDARの垂直解像度(分解能)

constexpr double CAMERA_RESOLUTION_H = (double)(((389/2) - 140) * LIDAR_RESOLUTION_H) / (double)304; // カメラの1[pix]あたりの角度[rad]
constexpr double CAMERA_RESOLUTION_V = (double)(((89/2) - 14) * LIDAR_RESOLUTION_V) / (double)193; // カメラの1[pix]あたりの角度[rad]

constexpr double DETECT_HEIGHT_MIN = 2; // 障害物の高さ(最小)
constexpr double DETECT_HEIGHT_MAX = 4; // 障害物の高さ(最大)
constexpr double LIDAR_HEIGHT = 0.97; // LiDARの取り付け高さ
constexpr double REFLECT_THRESH = 0.9f; // 反射強度の閾値
constexpr double MIN_RANGE = 5.0f; // 投影する距離の最小値
constexpr double MAX_RANGE = 15.0f; // 投影する距離の最大値
constexpr double MAX_REFLECT = 1.0f; // 投影する反射強度の最大値
constexpr double CLUSTERING_DISTANCE = 0.5; // 同一物体と判断する距離
constexpr double MIN_OBJECT_SIZE = 50; // 検出する物体の最小の画素数
constexpr int FEATURE_HOG_BIN_NUM = 60; // HOGのビンの数
constexpr int FEATURE_REF_BIN_NUM = 50; // 反射強度ヒストグラムのビンの数
constexpr double MATCH_THRESHOLD = 0.7; // マッチ判定の閾値
constexpr int OBSTACLE_BUFFER_SIZE = 2; // バッファとして確保する過去の障害物の数
constexpr int MIN_PIX_NUM_SIGN = 20; // 標識と判定するピクセル数の最小
constexpr int MAX_PIX_NUM_SIGN = 80; // 標識と判定するピクセル数の最大
constexpr double MIN_ASPECT_RATIO_SIGN = 0.2; // 標識と判定するアスペクト比の最小
constexpr double MAX_ASPECT_RATIO_SIGN = 0.8; // 標識と判定するアスペクト比の最大

// 赤青黃のHSV表色系での閾値を設定
constexpr int MIN_H_RED_01 = 1;
constexpr int MAX_H_RED_01 = 10;
constexpr int MIN_H_RED_02 = 165;
constexpr int MAX_H_RED_02 = 180;
constexpr int MIN_S_RED = 35;
constexpr int MAX_S_RED = 255;
constexpr int MIN_V_RED = 40;
constexpr int MAX_V_RED = 255;

constexpr int MIN_H_GREEN = 60;
constexpr int MAX_H_GREEN = 95;
constexpr int MIN_S_GREEN = 35;
constexpr int MAX_S_GREEN = 255;
constexpr int MIN_V_GREEN = 40;
constexpr int MAX_V_GREEN = 255;

constexpr int MIN_H_YELLOW = 0;
constexpr int MAX_H_YELLOW = 60;
constexpr int MIN_S_YELLOW = 40;
constexpr int MAX_S_YELLOW = 255;
constexpr int MIN_V_YELLOW = 145;
constexpr int MAX_V_YELLOW = 255;

// 信号が青なのか赤なのか判断するフラグ
bool green_light_flag = false;
bool red_light_flag = false;

// IMAGE_THRESHフレーム連続で赤、青が認識されると信号とみなす
constexpr int RED_IMAGE_THRESH = 0;
constexpr int GREEN_IMAGE_THRESH = 0;

// 赤、青信号が何フレーム連続で検出されたか数えるcount
int red_cnt = 0;
int green_cnt = 0;

// 赤or青判定を画像に表示する文字
std::string light_msg_state;

// 信号の候補領域のピクセル数の閾値
int pixel_num = 0;
constexpr int MIN_PIX_NUM = 150;
constexpr int MAX_PIX_NUM = 1200;

// 信号の候補領域のアスペクト比の閾値
// 横 : 縦 = ASPECT_RATIO : 1
double aspect_ratio = .0f;
constexpr double MIN_ASPECT_RATIO = 0.6;
constexpr double MAX_ASPECT_RATIO = 1.6;

vector<TrackableObstacle> obstacles_buffer;
vector<vector<LidarData>> points_buffer;

/* ファイル名を取得 */
void getFiles(const fs::path &path, const string &extension, vector<fs::path> &files)
{
  for(const fs::directory_entry &p : fs::directory_iterator(path)){
    if(!fs::is_directory(p)){
      if(p.path().extension().string() == extension){
        files.push_back(p);
      }
    }
  }
  sort(files.begin(), files.end());
}
// pcdファイルを読み込む関数
bool loadPCD(const string &path, vector<LidarData> &points)
{
  if (pr::loadPointsFromLog(path, points)) {
      // for (const auto& point : points) {
      //     cout << "x: " << point.x << ", y: " << point.y << ", z: " << point.z << ", reflectivity: " << point.reflectivity << endl;
      // }
      return true;
  } else {
      cerr << "Failed to load points from file." << endl;
      return false;
  }
}

void euler2Quaternion(float roll, float pitch, float yaw, float &q_w, float &q_x, float &q_y, float &q_z)
{
  double cos_roll = cos(roll / 2.0);
  double sin_roll = sin(roll / 2.0);
  double cos_pitch = cos(pitch / 2.0);
  double sin_pitch = sin(pitch / 2.0);
  double cos_yaw = cos(yaw / 2.0);
  double sin_yaw = sin(yaw / 2.0);
  q_w = cos_roll * cos_pitch * cos_yaw + sin_roll * sin_pitch * sin_yaw;
  q_x = sin_roll * cos_pitch * cos_yaw - cos_roll * sin_pitch * sin_yaw;
  q_y = cos_roll * sin_pitch * cos_yaw + sin_roll * cos_pitch * sin_yaw;
  q_z = cos_roll * cos_pitch * sin_yaw - sin_roll * sin_pitch * cos_yaw;
}

void rotatePoints(const vector<LidarData> &src, float roll, float pitch, float yaw, vector<LidarData> &dst)
{
  int sz = src.size();
  float qw, qx, qy, qz;
  euler2Quaternion(roll, pitch, yaw, qw, qx, qy, qz);
  qz = -qz;
  if(dst.size() != src.size()) dst.resize(sz);
  vector<float> r{
    (qw * qw) + (qx * qx) - (qy * qy) - (qz * qz), 2 * (qw * qz + qx * qy), 2 * (qx * qz - qw * qy),
    2 * (qx * qy - qw * qz), qw * qw - qx * qx + qy * qy - qz * qz, 2 * (qy * qz + qw * qx),
    2 * (qw * qy + qx * qz), 2 * (-qw * qx + qy * qz), qw * qw - qx * qx - qy * qy + qz * qz
  };
  for(int i = 0; i < sz; i++){
    float x = src[i].x, y = src[i].y, z = src[i].z;
    dst[i].x = r[0] * x + r[1] * y + r[2] * z;
    dst[i].y = r[3] * x + r[4] * y + r[5] * z;
    dst[i].z = r[6] * x + r[7] * y + r[8] * z;

    dst[i].x = dst[i].x - X_DIFF;
    dst[i].y = dst[i].y - Y_DIFF;
    dst[i].z = dst[i].z - Z_DIFF;

    dst[i].range = src[i].range;
    dst[i].reflectivity = src[i].reflectivity;
  }
}

void projectToImage(const vector<LidarData> &points,Mat &lidar_img)
{
  // 視野角を解像度で割ると幅と高さのピクセル数が求まる
  int width = cvRound(LIDAR_VIEW_ANGLE_H/LIDAR_RESOLUTION_H);
  int height = cvRound(LIDAR_VIEW_ANGLE_V/LIDAR_RESOLUTION_V);
  lidar_img = Mat(height,width, CV_32FC2,Scalar(.0, .0));
  
  // points[i]の水平角度と垂直角度を求める
  int sz = points.size();
  for(int i=0; i<sz;i++)
  {
    if(points[i].z < DETECT_HEIGHT_MIN) continue;
    double angle_h = atan2(points[i].y,points[i].x); 
    double angle_v = atan2(points[i].z - LIDAR_HEIGHT,sqrt(points[i].x * points[i].x + points[i].y * points[i].y));
    // 求めた角度が画像のどこの画素に対応するか求める
    int x = cvRound(width/2 - angle_h/LIDAR_RESOLUTION_H);
    int y = cvRound((height/2 - angle_v/LIDAR_RESOLUTION_V));

    // cout << "points["<<i<<"].lidar_img_rfl_x : " << x <<endl;
    // cout << "points["<<i<<"].lidar_img_rfl_y : " << y <<endl;

    if(x<0 || y<0 || x>=width || y>=height) continue;
    lidar_img.at<Vec2f>(y,x) = Vec2f(points[i].reflectivity, points[i].range);
  }
  height = cvRound(height* DEG_TO_RAD / LIDAR_RESOLUTION_H);
  resize(lidar_img, lidar_img, Size(width, height), 0, 0, INTER_NEAREST);
}

void detect(const vector<LidarData> &points, TrackableObstacle &obstacle)
{
  projectToImage(points, obstacle.lidar_img);
}

void drawObjectsReflect(const TrackableObstacle &obstacle, Mat &img)
{
  auto remap = [](float val, float from_low, float from_high, float to_low, float to_high)
  {
    return (val - from_low) * (to_high - to_low) / (from_high - from_low) + to_low;
  };
  auto &lidar_img = obstacle.lidar_img;
  if(img.cols != lidar_img.cols || img.rows != lidar_img.rows) img = Mat(lidar_img.size(),CV_8UC3, Scalar(.0, .0, .0));
  else img = Scalar(.0, .0, .0);
  for(int y=0;y<img.rows; y++)
  {
    for(int x=0; x<img.cols; x++)
    {
      float reflect = lidar_img.at<Vec2f>(y,x)[0];
      float range = lidar_img.at<Vec2f>(y,x)[1];
      if(reflect>MAX_REFLECT || reflect<= .0f) continue; // 必須
      if(range>MAX_RANGE || range<= MIN_RANGE) continue; // 必須
      if(reflect<REFLECT_THRESH) continue; // 閾値処理
      int val = (int)remap(reflect, 0.0f, (float)MAX_REFLECT, 30.0f, 255.0f);
      img.at<Vec3b>(y,x) = Vec3b(val, val, val);
    }
  }
}

void drawObjectsRange(const TrackableObstacle &obstacle, Mat &img)
{
  auto remap = [](float val, float from_low, float from_high, float to_low, float to_high)
  {
    return (val - from_low) * (to_high - to_low) / (from_high - from_low) + to_low;
  };
  auto &lidar_img = obstacle.lidar_img;
  if(img.cols != lidar_img.cols || img.rows != lidar_img.rows) img = Mat(lidar_img.size(),CV_8UC3, Scalar(.0, .0, .0));
  else img = Scalar(.0, .0, .0);
  for(int y=0;y<img.rows; y++)
  {
    for(int x=0; x<img.cols; x++)
    {
      float range = lidar_img.at<Vec2f>(y,x)[1];
      if(range>MAX_RANGE || range<= MIN_RANGE) continue;
      int val = (int)remap(range, (float)MAX_RANGE, 0.0f, 30.0f, 255.0f);
      // int val = (int)remap(range, 0.0f, (float)MAX_RANGE, 30.0f, 255.0f);
      img.at<Vec3b>(y,x) = Vec3b(val, val, val);
    }
  }
}

void saveCroppedRectangle(const Mat &input_image, const Rect &roi,const string &directory_name, const string &file_name)
{
  Mat cropped_image = input_image(roi).clone();
  std::string output_path = directory_name + "/" + file_name;
  imwrite(output_path, cropped_image);
}

void rectangleReflect(const Mat &lidar_reflect_img, const Mat &camera_img, 
                      vector<int> &pts1_x_sign, vector<int> &pts1_y_sign, vector<int> &pts2_x_sign, vector<int> &pts2_y_sign,
                      vector<int> &pts1_x_region, vector<int> &pts1_y_region, vector<int> &pts2_x_region, vector<int> &pts2_y_region)
{
  Mat lidar_reflect_img_gray;
  Mat reflect_stats;
  Mat reflect_centroids;
  cv::cvtColor(lidar_reflect_img, lidar_reflect_img_gray, cv::COLOR_BGR2GRAY);
  std::vector<int> widths, heights, lefts, tops;
  int object_reflect_id = 0;

  object_reflect_id = cv::connectedComponentsWithStats(lidar_reflect_img_gray, lidar_reflect_img_gray, reflect_stats, reflect_centroids);
  for(int label = 0; label < object_reflect_id; label++)
  {
    int width = reflect_stats.at<int>(label, cv::CC_STAT_WIDTH);
    int height = reflect_stats.at<int>(label, cv::CC_STAT_HEIGHT);
    int left = reflect_stats.at<int>(label, cv::CC_STAT_LEFT);
    int top = reflect_stats.at<int>(label, cv::CC_STAT_TOP);

    widths.push_back(width);
    heights.push_back(height);
    lefts.push_back(left);
    tops.push_back(top);
    // 矩形を描く
    // ピクセル数とアスペクト比を見る
    int pixel_num = width * height;
    double aspect_ratio = ((double)width) / ((double)height);
    if(pixel_num<MIN_PIX_NUM_SIGN || pixel_num>MAX_PIX_NUM_SIGN || aspect_ratio<MIN_ASPECT_RATIO_SIGN || aspect_ratio>MAX_ASPECT_RATIO_SIGN)
    {
      continue;
    }
    cv::rectangle(lidar_reflect_img, cv::Rect(lefts[label], tops[label], widths[label], heights[label]), cv::Scalar(0, 255, 255), 0);
    
    // cout << "lidar_reflect_img.size().width" << lidar_reflect_img.size().width << endl;
    // cout << "lidar_reflect_img.size().height" << lidar_reflect_img.size().height << endl;
    // cout << "H方向角度" << (lidar_reflect_img.size().width - 140) * LIDAR_RESOLUTION_H << endl;
    // cout << "V方向角度" << (lidar_reflect_img.size().height - 14) * LIDAR_RESOLUTION_V << endl;
    // cout << "label : " << label << endl;

    int angle_h_pix = cvRound(((lidar_reflect_img.size().width)/2) - lefts[label]);
    int angle_v_pix = cvRound(((lidar_reflect_img.size().height)/2) - tops[label]);
    double angle_h = angle_h_pix * LIDAR_RESOLUTION_H;
    double angle_v = angle_v_pix * LIDAR_RESOLUTION_V;
    
    // cout << "lefts[" << label << "] : " << lefts[label] << endl;
    // cout << "tops[" << label << "] : " << tops[label] << endl;
    // cout << "widths[label] : " << widths[label] << endl;
    // cout << "heights[label] : " << heights[label] << endl;
    // cout << "lidar_reflect_img.size().width : " << lidar_reflect_img.size().width << endl;
    // cout << "lidar_reflect_img.size().height : " << lidar_reflect_img.size().height << endl;
    // cout << "angle_h_pix : " << angle_h_pix << endl;
    // cout << "angle_v_pix : " << angle_v_pix << endl;
    // cout << "angle_h[rad] : " << angle_h << endl; 
    // cout << "angle_v[rad] : " << angle_v << endl; 
    // cout << "angle_h[deg] : " << angle_h * (180/M_PI) << endl; 
    // cout << "angle_v[deg] : " << angle_v * (180/M_PI) << endl;

    int angle_camera_h_pix = cvRound(angle_h / CAMERA_RESOLUTION_H);
    int angle_camera_v_pix = cvRound(angle_v / CAMERA_RESOLUTION_V);

    // cout << "angle_camera_h_pix : " << angle_camera_h_pix << endl;
    // cout << "angle_camera_v_pix : " << angle_camera_v_pix << endl;

    int camera_left = camera_img.size().width / 2 - angle_camera_h_pix;
    int camera_top = camera_img.size().height / 2 - angle_camera_v_pix;
    int camera_width = width * (20/3);
    int camera_height = height * (48/8);

    // cout << "camera_left : " << camera_left << endl;
    // cout << "camera_top : " << camera_top << endl;
    // cout << "camera_width : " << camera_width << endl;
    // cout << "camera_height : " << camera_height << endl;

    int pt1_x_sign, pt1_y_sign, pt2_x_sign, pt2_y_sign;
    if(camera_left < 0){
      pt1_x_sign = 0;
    }else{
      pt1_x_sign = camera_left;
    }
    if(camera_top < 0){
      pt1_y_sign = 0;
    }else{
      pt1_y_sign = camera_top;
    }
    if(camera_left + camera_width > camera_img.size().width){
      pt2_x_sign = camera_img.size().width;
    }else{
      pt2_x_sign = camera_left + camera_width;
    }
    if(camera_top + camera_height > camera_img.size().height){
      pt2_y_sign = camera_img.size().height;
    }else{
      pt2_y_sign = camera_top + camera_height;
    }
    // よく分からないが、push_backしてから代入したら上手くいく。初期化が必要なのかも
    pts1_x_sign.push_back(pt1_x_sign);
    pts1_y_sign.push_back(pt1_y_sign);
    pts2_x_sign.push_back(pt2_x_sign);
    pts2_y_sign.push_back(pt2_y_sign);
    pts1_x_sign[label] = pt1_x_sign;
    pts1_y_sign[label] = pt1_y_sign;
    pts2_x_sign[label] = pt2_x_sign;
    pts2_y_sign[label] = pt2_y_sign;
    cv::Rect rect_sign(Point(pts1_x_sign[label], pts1_y_sign[label]), Point(pts2_x_sign[label], pts2_y_sign[label]));

    // cv::rectangle(camera_img, rect_sign, cv::Scalar(0,255,255), 2);

    // cv::Rect roi(lefts[label], tops[label], widths[label], heights[label]);
    // std::string directory_name = "/home/chiba/share/camera_lidar_data/tmp/000032";
    // std::string file_name = "cropped_image_" + std::to_string(label) + ".png";
    // saveCroppedRectangle(lidar_reflect_img, roi, directory_name, file_name);

    int pt1_x_region, pt1_y_region, pt2_x_region, pt2_y_region;
    if(camera_left - 2*camera_width < 0){
      pt1_x_region = 0;
    }else{
      pt1_x_region = camera_left - 2*camera_width;
    }
    if(camera_top - 0.3*camera_height < 0){
      pt1_y_region = 0;
    }else{
      pt1_y_region = camera_top - 0.3*camera_height;
    }
    if(camera_left + camera_width + 1*camera_width > camera_img.size().width){
      pt2_x_region = camera_img.size().width;
    }else{
      pt2_x_region = camera_left + camera_width + 1*camera_width;
    }
    if(camera_top + camera_height + 0.4*camera_height > camera_img.size().height * 0.4){
      pt2_y_region = camera_img.size().height * 0.4;
    }else{
      pt2_y_region = camera_top + camera_height + 0.4*camera_height;
    }
    pts1_x_region.push_back(pt1_x_region);
    pts1_y_region.push_back(pt1_y_region);
    pts2_x_region.push_back(pt2_x_region);
    pts2_y_region.push_back(pt2_y_region);
    pts1_x_region[label] = pt1_x_region;
    pts1_y_region[label] = pt1_y_region;
    pts2_x_region[label] = pt2_x_region;
    pts2_y_region[label] = pt2_y_region;
    cv::Rect rect_region(Point(pts1_x_region[label], pts1_y_region[label]), Point(pts2_x_region[label], pts2_y_region[label]));
    // cv::rectangle(camera_img, rect_region, Scalar(255, 0, 110), 2);
  }
}

void saveRangeImage(const Mat &lidar_range_img)
{
  string directory_path = "/home/chiba/share/camera_lidar_data/tmp/000000";
  string file_path = "range_tmp.png";
  string output_path = directory_path + "/" + file_path;
  imwrite(output_path, lidar_range_img);
}

// ここから信号認識用関数 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/* カメラ画像から赤色の画素を抽出する関数 */
void extractRedSignal(cv::Mat &rgb, cv::Mat &hsv, cv::Mat &extract_red)
{
  // 赤信号の赤色を抽出
  for(int y = 0; y < rgb.rows; y++){
      for(int x = 0; x < rgb.cols; x++){
          cv::Vec3b val = hsv.at<cv::Vec3b>(y, x);
          if(    /*(MIN_H_RED_01 <= val[0] && val[0] <= MAX_H_RED_01) ||*/
                 (MIN_H_RED_02 <= val[0] && val[0] <= MAX_H_RED_02)
              && MIN_S_RED <= val[1] && val[1] <= MAX_S_RED
              && MIN_V_RED <= val[2] && val[2] <= MAX_V_RED)
          {
            extract_red.at<cv::Vec3b>(y, x) = rgb.at<cv::Vec3b>(y, x);
          }
      }
  }
  cv::imshow("extract_red",extract_red);
}

/* カメラ画像から緑色の画素を抽出する関数 */
void extractGreenSignal(cv::Mat &rgb, cv::Mat &hsv, cv::Mat &extract_green)
{
  // 青信号の緑色を抽出
  for(int y = 0; y < rgb.rows; y++){
      for(int x = 0; x < rgb.cols; x++){
          cv::Vec3b val = hsv.at<cv::Vec3b>(y,x);
          if(    MIN_H_GREEN <= val[0] && val[0] <= MAX_H_GREEN
              && MIN_S_GREEN <= val[1] && val[1] <= MAX_S_GREEN
              && MIN_V_GREEN <= val[2] && val[2] <= MAX_V_GREEN){
              extract_green.at<cv::Vec3b>(y,x) = rgb.at<cv::Vec3b>(y,x);
          }
      }
  }
  cv::imshow("extract_green",extract_green);
}

/* 抽出した色を白くし、二値化する関数 */
void binalizeImage(cv::Mat &src, cv::Mat &gray_img)
{
    for(int y = 0; y<src.rows; y++)
    {
        for(int x = 0; x<src.cols; x++)
        {
            if(src.at<cv::Vec3b>(y,x)!=cv::Vec3b(0, 0, 0))
            {
                gray_img.at<uchar>(y,x) = 255;
            }
        }
    }
}

/* ピンク色または水色の中に黄色が見えたら赤or青色の矩形でラベリングする関数 */
/* isRedLightがtrueなら赤信号用の処理、falseなら青信号用の処理になる */
void extractYellowInBlob(cv::Mat &rgb, cv::Mat &bin_img, int num_labels, const std::vector<int> &widths, const std::vector<int> &heights, const std::vector<int> &lefts, const std::vector<int> &tops, bool isRedSignal,
                          const vector<int> &pts1_x_region, const vector<int> &pts1_y_region, const Mat &region_img, int region_num)
{
  cv::Mat hsv;
  cv::cvtColor(rgb, hsv, cv::COLOR_BGR2HSV);
  for (int label = 0; label < num_labels; label++)
  {
    int left = lefts[label];
    int top = tops[label];
    int width = widths[label];
    int height = heights[label];

    // ピクセル数とアスペクト比を見る
    pixel_num = height * width;
    aspect_ratio = ((double)width)/((double)height);
    if(pixel_num<MIN_PIX_NUM || pixel_num>MAX_PIX_NUM)
    {
      continue;
    }
    if(aspect_ratio<MIN_ASPECT_RATIO || aspect_ratio>MAX_ASPECT_RATIO)
    {
      continue;
    }

    cv::Mat blob_rgb(rgb, cv::Rect(left + pts1_x_region[region_num], top + pts1_y_region[region_num], width, height));
    cv::imshow("rgb", blob_rgb);
    cv::Mat blob_hsv(hsv, cv::Rect(left + pts1_x_region[region_num], top + pts1_y_region[region_num], width, height));
    // cv::imshow("blob_hsv",blob_hsv);

    cv::Mat extract_yellow = cv::Mat::zeros(blob_rgb.size(), blob_rgb.type());
    cv::medianBlur(blob_hsv, blob_hsv, 3);
    for (int y = 0; y < blob_hsv.rows; y++)
    {
      for (int x = 0; x < blob_hsv.cols; x++)
      {
        cv::Vec3b val = blob_hsv.at<cv::Vec3b>(y, x);
        if (   (MIN_H_YELLOW <= val[0] && val[0] <= MAX_H_YELLOW)
            && (MIN_S_YELLOW <= val[1] && val[1] <= MAX_S_YELLOW)
            && (MIN_V_YELLOW <= val[2] && val[2] <= MAX_V_YELLOW))
        {
          extract_yellow.at<cv::Vec3b>(y, x) = blob_rgb.at<cv::Vec3b>(y, x);
        }
      }
    }

    cv::imshow("extract_yellow", extract_yellow);
    cv::Mat bin_img_yellow = cv::Mat::zeros(blob_hsv.size(), CV_8UC1);
    binalizeImage(extract_yellow, bin_img_yellow);

    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    // cv::Mat eroded, dilated;
    // cv::erode(bin_img_yellow, eroded, kernel);
    // cv::dilate(eroded, dilated, kernel);

    cv::Mat labeled_yellow;
    cv::Mat stats_yellow, centroids_yellow;
    int num_labels_yellow = cv::connectedComponentsWithStats(bin_img_yellow, labeled_yellow, stats_yellow, centroids_yellow);
    for (int label = 0; label < num_labels_yellow; label++)
    {
      int yellow_width = stats_yellow.at<int>(label, cv::CC_STAT_WIDTH);
      int yellow_height = stats_yellow.at<int>(label, cv::CC_STAT_HEIGHT);
      int yellow_left = stats_yellow.at<int>(label, cv::CC_STAT_LEFT);
      int yellow_top = stats_yellow.at<int>(label, cv::CC_STAT_TOP);

      // cv::rectangle(bin_img_yellow, cv::Rect(yellow_left, yellow_top, yellow_width, yellow_height), cv::Scalar(256/2), 2);
      if (isRedSignal)
      {
        if(num_labels_yellow > 1){
          cv::rectangle(rgb, cv::Rect(left + pts1_x_region[region_num], top + pts1_y_region[region_num], width, height), cv::Scalar(0, 0, 255), 2); // 赤信号は赤い矩形
          // cv::rectangle(region_img, cv::Rect(left, top, width, height),cv::Scalar(0, 0, 255), 2);
          red_light_flag = true;
        }
      }
      else
      {
        if(num_labels_yellow > 1){
          cv::rectangle(rgb, cv::Rect(left + pts1_x_region[region_num], top + pts1_y_region[region_num], width, height), cv::Scalar(255, 0, 0), 2); // 青信号は青い矩形
          // cv::rectangle(region_img, cv::Rect(left, top, width, height),cv::Scalar(255, 0, 0), 2);
          green_light_flag = true;
        }
      }
    }

    for (int y = 0; y < bin_img_yellow.rows; y++)
    {
      for (int x = 0; x < bin_img_yellow.cols; x++)
      {
        if (bin_img_yellow.at<uchar>(y, x) != 0)
        {
          bin_img.at<uchar>(top + y, left + x) = 255;
        }
      }
    }
    // cv::imshow("nin_img_yellow",bin_img_yellow);
    // cv::imshow("bin_img",bin_img);
  }
}

void drawOverlay(cv::Mat &image, bool red_light_flag, bool green_light_flag) 
{
  cv::Scalar color;
  if (red_light_flag) {
    color = cv::Scalar(0, 0, 255); // 赤色
    cv::rectangle(image, cv::Rect(10, 580, 60, 60), color, -1); // 赤信号の位置に赤い塗りつぶし矩形を描画
  } 
  if(green_light_flag) {
    color = cv::Scalar(255, 0, 0); // 青色
    cv::rectangle(image, cv::Rect(10, 650, 60, 60), color, -1); // 青信号の位置に青い塗りつぶし矩形を描画
  }
}

void addTextToImage(cv::Mat &image, const std::string &light_msg_state)
{
  int fontFace = cv::FONT_HERSHEY_SIMPLEX;
  double fontScale = 2;
  int thickness = 5;
  cv::Point textOrg(0, 0);
  if(light_msg_state=="RedLight")
  {
    cv::Point textOrg(80, 630); // 文字列を表示する位置
    cv::putText(image, light_msg_state, textOrg, fontFace, fontScale, cv::Scalar(0, 0, 255), thickness);
  } else if(light_msg_state=="GreenLight"){
    cv::Point textOrg(80, 700); // 文字列を表示する位置
    cv::putText(image, light_msg_state, textOrg, fontFace, fontScale, cv::Scalar(255, 0, 0), thickness);
  }
}
// ここまで信号認識用関数 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc,char **argv){
  vector<LidarData> points;
  vector<fs::path> files_png;
  vector<fs::path> files_pcd;
  Mat lidar_img;
  Mat lidar_range_img;
  Mat lidar_reflect_img;

  /*** キャリブレーションパラメータの読み込み ***/
  // カメラ画像の歪み補正
  Mat camera_params, distortion_params, rectify_params, projection_params;
  Mat map_x, map_y;
  FileStorage fs(argv[3], FileStorage::READ);
  if(!fs.isOpened()) return false;
  Size sz((int)fs["image_width"], (int)fs["image_height"]);
  auto model = (string)fs["distortion_model"];
  fs["camera_matrix"] >> camera_params;
  fs["distortion_coefficients"] >> distortion_params;
  fs["rectification_matrix"] >> rectify_params;
  fs["projection_matrix"] >> projection_params;
  initUndistortRectifyMap(camera_params, distortion_params, rectify_params, projection_params, sz, CV_32FC1, map_x, map_y);

  double mgn = 2; // magnification : 倍率

  if(argc < 4){
    cout << "Usage is : " <<  argv[0] << "[image_directory_path][pcd_directory_path][camera_distotion_parameter.yaml]" << endl;
    return -1;
  }
  /* ディレクトリからpngおよびpcdファイル名を取得 */
  /* files_pngおよびfiles_pcdにファイル名を格納 */
  getFiles(argv[1],".png",files_png);
  getFiles(argv[2],".pcd",files_pcd);

  int file_cnt = 0;
  int now_obs_cnt_sum = 0;

  while (true)
  {
    vector<LidarData> src_points;
    vector<LidarData> points;
    TrackableObstacle obstacle;
    Mat obstacle_now;
    Mat obstacle_past;
    vector<int> pts1_x_sign, pts1_y_sign, pts2_x_sign, pts2_y_sign;
    vector<int> pts1_x_region, pts1_y_region, pts2_x_region, pts2_y_region;

    /* カメラ画像の読み込み */
    Mat src_camera_img = imread(files_png[file_cnt].string(),1);
    Mat camera_img;
    if (src_camera_img.empty())
    {
      std::cerr << "Error: Could not read camera_img" << std::endl;
      return -1;
    }
    cout << "camera file : " << files_png[file_cnt].string() << endl;
    /*** 変換処理、カメラ画像の歪み補正 ***/
    remap(src_camera_img, camera_img, map_x, map_y, INTER_LINEAR);
    // std::string saveDirectory = "/home/chiba/";
    // imwrite(saveDirectory+ "000000.png", camera_img);

    /* 点群の読み込み */
    loadPCD(files_pcd[file_cnt].string(), src_points);
    /*** 点群をカメラ画像の座標に合わせる(回転させる)関数 ***/
    rotatePoints(src_points, degToRad(ROLL), degToRad(PITCH), degToRad(YAW), points);
    /*** 反射強度画像，距離画像の作成 ***/
    detect(points, obstacle);

    int now_obs_cnt;

    // drawCorresponding(obstacle_now, obstacle_past, now_obs_cnt);
    now_obs_cnt_sum += now_obs_cnt;

    drawObjectsReflect(obstacle, lidar_reflect_img);
    drawObjectsRange(obstacle, lidar_range_img);
    rectangleReflect(lidar_reflect_img, camera_img,
                      pts1_x_sign, pts1_y_sign, pts2_x_sign, pts2_y_sign,
                      pts1_x_region, pts1_y_region, pts2_x_region, pts2_y_region);

    // saveRangeImage(lidar_range_img);

    // resize(obstacle_now, obstacle_now, Size(), mgn, mgn);
    // resize(obstacle_past, obstacle_past, Size(), mgn, mgn);

    for(int region_num = 0; region_num < pts1_x_region.size(); region_num++)
    {
      cv::Rect region(Point(pts1_x_region[region_num], pts1_y_region[region_num]), Point(pts2_x_region[region_num], pts2_y_region[region_num]));
      cv::Mat region_img = camera_img(region).clone();

      cv::imshow("region_img", region_img);

      cv::Mat hsv;
      cv::cvtColor(region_img, hsv, cv::COLOR_BGR2HSV);

      cv::Mat extract_red(region_img.size(), region_img.type(), cv::Scalar(0, 0, 0));
      cv::Mat extract_green(region_img.size(), region_img.type(), cv::Scalar(0, 0, 0));

      // カメラ画像から赤緑を抽出
      extractRedSignal(region_img, hsv, extract_red);
      extractGreenSignal(region_img, hsv, extract_green);
 

      // メディアンフィルターにかける
      cv::Mat red_median(region_img.size(), region_img.type(), cv::Scalar(0, 0, 0));
      cv::Mat green_median(region_img.size(), region_img.type(), cv::Scalar(0, 0, 0));
      cv::medianBlur(extract_red, red_median, 3);
      cv::medianBlur(extract_green, green_median, 3);

      // 二値化
      cv::Mat bin_img_red = cv::Mat::zeros(region_img.size(), CV_8UC1);
      cv::Mat bin_img_green = cv::Mat::zeros(region_img.size(), CV_8UC1);
      binalizeImage(red_median, bin_img_red);
      binalizeImage(green_median, bin_img_green);

      // // 収縮処理
      // cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
      // cv::erode(bin_img_red, bin_img_red, kernel_erode);
      // cv::erode(bin_img_green, bin_img_green, kernel_erode);
      // 膨張処理
      cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
      cv::dilate(bin_img_red, bin_img_red, kernel_dilate);
      cv::dilate(bin_img_green, bin_img_green, kernel_dilate);

      cv::imshow("bin_img_red", bin_img_red);
      cv::imshow("bin_img_green", bin_img_green);

      // ラベリング
      cv::Mat labeled_red, labeled_green;
      cv::Mat stats_red, states_green, centroids_red, centroids_green;
      int num_labels_red, num_labels_green;

      num_labels_red = cv::connectedComponentsWithStats(bin_img_red, labeled_red, stats_red, centroids_red);
      std::vector<int> red_width, red_height, red_left, red_top;
      for (int label = 0; label < num_labels_red; label++)
      {
        int left = stats_red.at<int>(label, cv::CC_STAT_LEFT);
        int top = stats_red.at<int>(label, cv::CC_STAT_TOP);
        int width = stats_red.at<int>(label, cv::CC_STAT_WIDTH);
        int height = stats_red.at<int>(label, cv::CC_STAT_HEIGHT);

        red_left.push_back(left);
        red_top.push_back(top);
        red_width.push_back(width);
        red_height.push_back(height);

        // cv::rectangle(bin_img_red, cv::Rect(left, top, width, height), cv::Scalar(256/2), 2);
        // ピンク色の矩形を描く
        // ピクセル数とアスペクト比を見る
        pixel_num = width * height;
        // std::cout << "pixel_num_red : " << pixel_num << std::endl;
        aspect_ratio = ((double)width)/((double)height);
        // std::cout << "aspect_ratio_red : " << aspect_ratio << std::endl;
        if(pixel_num<MIN_PIX_NUM || pixel_num>MAX_PIX_NUM)
        {
          continue;
        }
        if(aspect_ratio<MIN_ASPECT_RATIO || aspect_ratio>MAX_ASPECT_RATIO)
        {
          continue;
        }
        cv::rectangle(camera_img, cv::Rect(red_left[label] + pts1_x_region[region_num], red_top[label] + pts1_y_region[region_num], red_width[label], red_height[label]), cv::Scalar(255,0,255), 2);
        // cv::rectangle(region_img, cv::Rect(red_left[label], red_top[label], red_width[label], red_height[label]), cv::Scalar(255,0,255), 2);
        // cout << "pts1_x_region[" << region_num << "]: " << pts1_x_region[region_num] << endl;
        // cout << "pts1_y_region[" << region_num << "]: " << pts1_y_region[region_num] << endl;
      }

      num_labels_green = cv::connectedComponentsWithStats(bin_img_green, labeled_green, states_green, centroids_green);
      std::vector<int> green_width, green_height, green_left, green_top;
      for (int label = 0; label < num_labels_green; label++)
      {
        int width = states_green.at<int>(label, cv::CC_STAT_WIDTH);
        int height = states_green.at<int>(label, cv::CC_STAT_HEIGHT);
        int left = states_green.at<int>(label, cv::CC_STAT_LEFT);
        int top = states_green.at<int>(label, cv::CC_STAT_TOP);

        green_width.push_back(width);
        green_height.push_back(height);
        green_left.push_back(left);
        green_top.push_back(top);

        // cv::rectangle(bin_img_green, cv::Rect(left, top, width, height), cv::Scalar(256/2), 2);
        // 水色の矩形を描く
        // ピクセル数、アスペクト比を見る
        pixel_num = width * height;
        // cout << "pixel_num_green : " << pixel_num << std::endl;
        aspect_ratio = ((double)width) / ((double)height);
        // cout << "aspect_ratio_green : " << aspect_ratio << std::endl;
        if(pixel_num<MIN_PIX_NUM || pixel_num>MAX_PIX_NUM)
        {
          continue;
        }
        if(aspect_ratio<MIN_ASPECT_RATIO || aspect_ratio>MAX_ASPECT_RATIO)
        {
          continue;
        }
        cv::rectangle(camera_img, cv::Rect(green_left[label] + pts1_x_region[region_num], green_top[label] + pts1_y_region[region_num], green_width[label], green_height[label]), cv::Scalar(255,255,0), 2);
        // cv::rectangle(region_img, cv::Rect(green_left[label], green_top[label], green_width[label], green_height[label]), cv::Scalar(255,255,0), 2);
      }

      extractYellowInBlob(camera_img, bin_img_red, num_labels_red, red_width, red_height, red_left, red_top, true,
                            pts1_x_region, pts1_y_region, region_img, region_num);
      extractYellowInBlob(camera_img, bin_img_green, num_labels_green, green_width, green_height, green_left, green_top, false,
                            pts1_x_region, pts1_y_region, region_img, region_num);

      // 赤、青信号が連続で検出されるほどcountが加算されていく
      // red_light_flag, greem_light_flagはextractYellowBlob関数から出力されている
      if(red_light_flag)
      {
        ++red_cnt;
      }
      else
      {
        red_cnt = 0;
      }
      if(green_light_flag)
      {
        ++green_cnt;
      }
      else
      {
        green_cnt = 0;
      }
      if(red_cnt>RED_IMAGE_THRESH)
      {
        drawOverlay(camera_img, red_light_flag, green_light_flag);
        light_msg_state = "RedLight";
        addTextToImage(camera_img, light_msg_state);
        red_cnt=0;
      }
      if(green_cnt>GREEN_IMAGE_THRESH)
      {
        drawOverlay(camera_img, red_light_flag, green_light_flag);
        light_msg_state = "GreenLight";
        addTextToImage(camera_img, light_msg_state);
        green_cnt = 0;
      }

      red_light_flag = false;
      green_light_flag = false;
      cv::imshow("region_img", region_img);
    }

    
    imshow("camera_img", camera_img);
    imshow("lidar_range_img", lidar_range_img);
    imshow("lidar_reflect_img", lidar_reflect_img);
    // imshow("obstacle now", obstacle_now);
    // imshow("obstacle past", obstacle_past);
    int key = waitKey(0);
    if(key == ' ') break;
    else if(key == 'a') --file_cnt;
    else if(key == 'A') file_cnt -= 10;
    else if(key == 'D') file_cnt += 10;
    else if(key == 'd') ++file_cnt;
    if(file_cnt < 0) file_cnt = 0;
  }
  return 0;
}