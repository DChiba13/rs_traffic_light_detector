#include <iostream>
#include <experimental/filesystem>
#include <ryusei/common/logger.hpp>
#include <ryusei/common/defs.hpp>
#include <fstream>


using namespace project_ryusei;
using namespace cv;
using namespace std;
namespace pr = project_ryusei;
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
namespace fs = std::experimental::filesystem;

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

constexpr double CAMERA_RESOLUTION_H = (16.56 * DEG_TO_RAD) / 216; // カメラの1[pix]あたりの角度[rad]
constexpr double CAMERA_RESOLUTION_V = (24 * DEG_TO_RAD) / 174; // カメラの1[pix]あたりの角度[rad]

constexpr double DETECT_HEIGHT_MIN = 2; // 障害物の高さ(最小)
constexpr double DETECT_HEIGHT_MAX = 4; // 障害物の高さ(最大)
constexpr double LIDAR_HEIGHT = 0.97; // LiDARの取り付け高さ
constexpr double REFLECT_THRESH = 0.9f; // 反射強度の閾値
constexpr double MIN_RANGE = 10.0f; // 投影する距離の最小値
constexpr double MAX_RANGE = 15.0f; // 投影する距離の最大値
constexpr double MAX_REFLECT = 1.0f; // 投影する反射強度の最大値
constexpr double CLUSTERING_DISTANCE = 0.5; // 同一物体と判断する距離
constexpr double MIN_OBJECT_SIZE = 50; // 検出する物体の最小の画素数
constexpr int FEATURE_HOG_BIN_NUM = 60; // HOGのビンの数
constexpr int FEATURE_REF_BIN_NUM = 50; // 反射強度ヒストグラムのビンの数
constexpr double MATCH_THRESHOLD = 0.7; // マッチ判定の閾値
constexpr int OBSTACLE_BUFFER_SIZE = 2; // バッファとして確保する過去の障害物の数
constexpr int MIN_PIX_NUM = 20; // 標識と判定するピクセル数の最小
constexpr int MAX_PIX_NUM = 80; // 標識と判定するピクセル数の最大
constexpr double MIN_ASPECT_RATIO = 0.2; // 標識と判定するアスペクト比の最小
constexpr double MAX_ASPECT_RATIO = 0.8; // 標識と判定するアスペクト比の最大

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
    double angle_v = atan2(points[i].z - LIDAR_HEIGHT,sqrt(points[i].x*points[i].x + points[i].y*points[i].y));
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

int provisionalLabeling(const Mat &src, Mat &clusters, vector<pair<ushort, ushort>> &id_pairs)
{
  // 左上:val[0], 上:val[1]、右上:[2]、左:val[3]
  ushort id = 0;
  ushort min_val;
  float range;
  clusters = Mat(src.size(), CV_16UC1, Scalar(0xffff));
  int min_label_idx;

  for(int y=0;y<src.rows;y++)
  {
    for(int x=0;x<src.cols;x++)
    {
      if(src.at<Vec2f>(y,x)[1]>MAX_RANGE || src.at<Vec2f>(y,x)[1]<=.0f) continue;
      vector<ushort> vals={0xffff, 0xffff, 0xffff, 0xffff};
      // 例外処理　画像の端を除く
      if(y>0){
        if(x>0) vals[0] = clusters.at<ushort>(y-1, x-1);
        vals[1] = clusters.at<ushort>(y-1, x);
        if(x < clusters.cols-1) vals[2] = clusters.at<ushort>(y-1, x+1);
      } 
      if(x>0) vals[3] = clusters.at<ushort>(y, x-1);
      min_val = *min_element(vals.begin(), vals.end());
      /*** min_val=0xffffの場合、注目画素のラベルをidにする ***/
      if(min_val == 0xffff)
      {
        clusters.at<ushort>(y, x) = id++;
      }
      /*** min_val!=0xffffの場合、4近傍画素の内、最も距離が近いラベル値を代入 ***/
      else
      {
        float range = src.at<Vec2f>(y, x)[1]; // 注目画素の距離
        vector<float> ranges(4, 65535.0); // 4近傍画素の距離
        /*** rangesに"src"から取得した距離情報を代入 ***/
        /*** 例外処理 ***/
        if(y > 0){
          if(x > 0) ranges[0] = src.at<Vec2f>(y - 1, x - 1)[1];
          ranges[1] = src.at<Vec2f>(y - 1, x)[1];
          if(x < src.cols - 1) ranges[2] = src.at<Vec2f>(y - 1, x + 1)[1];
        }
        if(x > 0) ranges[3] = src.at<Vec2f>(y, x - 1)[1];
        /*** 注目画素との距離の差がCLUSTERING_DISTANCE_以下で、ラベル(vals)の値が最小となる画素を探索 ***/
        min_label_idx = -1; // ラベルの値が最小となる画素のインデックス(0〜3)
        min_val = 0xffff;
        for(int i = 0; i < vals.size(); i++){
          if(abs(range - ranges[i]) <= CLUSTERING_DISTANCE && vals[i] < min_val){
            min_label_idx = i;
            min_val = vals[i];
          }
        }
        /*** ↑で求めた画素のラベル値が初期値(0xffff)以外であれば注目画素にそのラベル値を代入 ***/
        if(min_label_idx >= 0 && min_label_idx < vals.size() && vals[min_label_idx] != 0xffff)
        {
          clusters.at<ushort>(y, x) = vals[min_label_idx];
          /*** 4近傍画素を探索して注目画素との距離の差がCLUSTERING_DISTANCE_以下でvals[min_label_idx]と異なる値がある場合はIDの振り直しリスト(id_pairs)を更新 ***/
          for(int i = 0; i < vals.size(); i++){
            if(abs(range - ranges[i]) <= CLUSTERING_DISTANCE && vals[i] != vals[min_label_idx])
            {
              //一度pairしたものは除外する←findを使う
              pair<ushort, ushort> id_pair(vals[i], vals[min_label_idx]);
              if(find(id_pairs.begin(), id_pairs.end(), id_pair) == id_pairs.end()) id_pairs.push_back(id_pair);
            }
          }
        }
        /*** ↑で求めた画素のラベル値が初期値(0xffff)の場合、注目画素のラベルをidにする ***/
        else //４近傍すべてCLUSTERING_DISTANCE_以上だったとき(遠かったとき)
        {
          clusters.at<ushort>(y, x) = id++;
        }
      }
    }
  }
  return id;
}

int integrateLabel(const vector<pair<ushort, ushort>> id_pairs, Mat &clusters, int max_id)
{
  /* ルックアップテーブルをclustersのラベルへ適用 */
  vector<int> d(max_id, 0);
  for(int i=0,size=id_pairs.size();i<size;++i)
  {
    if(id_pairs[i].second - id_pairs[i].first < 0) continue;
    d[id_pairs[i].first] = id_pairs[i].second - id_pairs[i].first;
  }
  int new_max_id=-1;
  for(int y=0;y<clusters.rows;y++)
  {
    for(int x=0;x<clusters.cols;x++)
    {
      auto &p = clusters.at<ushort>(y,x);
      if(p>=max_id) continue;
      p=p+d[p];
      if(p>new_max_id) new_max_id=p;
    }
  }
  return new_max_id;
}

int relabel(Mat &clusters, int max_id)
{
  vector<int> counts(max_id + 1, 0);
  /* それぞれのラベルの画素が何画素あるかcounts変数を使って数える */
  for(int y=0; y<clusters.rows; y++)
  {
    for(int x=0; x<clusters.cols; x++)
    {
      auto &p = clusters.at<ushort>(y,x);
      if(p>max_id) continue;
      ++counts[p];
    }
  }
  /* ↑でカウントした画素数がMIN_OBJECT_SIZE変数未満の場合は0xffffに変更 */
  vector<int> diffs(max_id+1, 0);
  int new_max_id = 0;
  int n=0;
  for(int i=0; i<counts.size();i++)
  {
    if(counts[i] >= MIN_OBJECT_SIZE)
    {
      diffs[i]=(i-n);
      n++;
    }else diffs[i]=0;
  }
  for(int y=0; y<clusters.rows; y++)
  {
    for(int x=0; x<clusters.cols; x++)
    {
      auto &p = clusters.at<ushort>(y,x);
      if(p>max_id) continue;
      if(counts[p] < MIN_OBJECT_SIZE)
      {
        p=0xffff;
        continue;
      } else{
        p=p-diffs[p];
      }
      if(p>new_max_id)
      {
        new_max_id = p;
      }
    }
  }
  return new_max_id;
}

void extractRectangles(const Mat &clusters, int cluster_num, vector<Rect> &rects)
{
  /* それぞれのラベルの矩形領域を算出し、rects変数に代入する */
  rects.resize(cluster_num + 1);
  vector<Point> ul(cluster_num + 1, Point(clusters.cols, clusters.rows)); // ul : upper left
  vector<Point> br(cluster_num + 1, Point(-1, -1)); // br : bottom right
  for(int y=0; y<clusters.rows; y++)
  {
    for(int x=0; x<clusters.cols; x++)
    {
      auto &p = clusters.at<ushort>(y,x);
      if(p>cluster_num) continue;
      if(ul[p].x > x) {
        ul[p].x = x;
        // cout << "ul[" << p << "].x : " << ul[p].x << endl;
      }
      if(ul[p].y > y) {
        ul[p].y = y;
        // cout << "ul[" << p << "].y : " <<  ul[p].y << endl;
      }
      if(br[p].x < x) {
        br[p].x = x;
        // cout << "br[" << p << "].x : " << br[p].x << endl;
      }
      if(br[p].y < y) {
        br[p].y = y;
        // cout << "br[" << p << "].y : " << br[p].y << endl;
      }
    }
  }
  for(int i=0; i<rects.size(); i++)
  {
    rects[i] = Rect(ul[i], br[i]);
  }
}

// 障害物を立方体として考え、立方体の8点と辺の長さを格納する関数
void rectanglesToObstacle(const TrackableObstacle &obstacle, vector<Obstacle> &obstacles)
{
  int sz = obstacle.rectangles.size();
  obstacles.resize(sz);
  vector<pair<float, float>> x_ranges(sz), y_ranges(sz), z_ranges(sz);
  for(int i=0; i<sz; i++)
  {
    x_ranges[i].first = y_ranges[i].first = z_ranges[i].first = MAX_RANGE * 10;
    x_ranges[i].second = y_ranges[i].second = z_ranges[i].second = -MAX_RANGE * 10;
  }

  auto &clusters = obstacle.clusters;
  int cx = clusters.cols/2, cy = clusters.rows/2;
  LidarData pt;
  for(int y=0; y<clusters.rows; y++)
  {
    for(int x=0; x<clusters.cols; x++)
    {
      auto &c = clusters.at<ushort>(y,x);
      if(c>=sz) continue;
      auto &l = obstacle.lidar_img.at<Vec2f>(y,x);
      double h = (cx - x) * LIDAR_RESOLUTION_H;
      double v = (cy - y) * LIDAR_RESOLUTION_H; // LIDAR_RESOLUTION_Vでは？
      pt.z = (l[1] * sinf(v)) + LIDAR_HEIGHT;
      float tmp_xy = l[1] * cosf(v);
      pt.x = tmp_xy * cosf(h);
      pt.y = tmp_xy * sinf(h);
      obstacles[c].points.push_back(LidarData(pt));
      if(x_ranges[c].second < pt.x) x_ranges[c].second = pt.x;
      if(x_ranges[c].first > pt.x) x_ranges[c].first = pt.x;
      if(y_ranges[c].second < pt.y) y_ranges[c].second = pt.y;
      if(y_ranges[c].first > pt.y) y_ranges[c].first = pt.y;
      if(z_ranges[c].second < pt.z) z_ranges[c].second = pt.z;
      if(z_ranges[c].first > pt.z) z_ranges[c].first = pt.z;
    }
  }
  for(int i = 0; i < sz; i++)
  {
    obstacles[i].relative_points.resize(8);
    obstacles[i].relative_points[0] = Point3f(x_ranges[i].first, y_ranges[i].second, z_ranges[i].second);
    obstacles[i].relative_points[1] = Point3f(x_ranges[i].first, y_ranges[i].first, z_ranges[i].second);
    obstacles[i].relative_points[2] = Point3f(x_ranges[i].second, y_ranges[i].first, z_ranges[i].second);
    obstacles[i].relative_points[3] = Point3f(x_ranges[i].second, y_ranges[i].second, z_ranges[i].second);
    obstacles[i].relative_points[4] = Point3f(x_ranges[i].first, y_ranges[i].second, z_ranges[i].first);
    obstacles[i].relative_points[5] = Point3f(x_ranges[i].first, y_ranges[i].first, z_ranges[i].first);
    obstacles[i].relative_points[6] = Point3f(x_ranges[i].second, y_ranges[i].first, z_ranges[i].first);
    obstacles[i].relative_points[7] = Point3f(x_ranges[i].second, y_ranges[i].second, z_ranges[i].first);
    obstacles[i].depth = abs(x_ranges[i].second - x_ranges[i].first);
    obstacles[i].width = abs(y_ranges[i].second - y_ranges[i].first);
    obstacles[i].height = abs(z_ranges[i].second - z_ranges[i].first);
  }
}

void calculateHog(const Mat &cluster, int id, vector<float> &hog)
{
  // cout << "id : "<< id <<endl;
  auto remap = [](float val, float from_low, float from_high, float to_low, float to_high)
  {
    return (val - from_low) * (to_high - to_low) / (from_high - from_low) + to_low;
  };
  hog.resize(FEATURE_HOG_BIN_NUM, .0f);
  Mat pad(cluster.rows + 4, cluster.cols + 4, CV_32FC1, Scalar(0));
  Mat pad_partial = pad(Rect(2,2,cluster.cols, cluster.rows));
  for(int y=0; y<cluster.rows; y++)
  {
    for(int x=0; x<cluster.cols; x++)
    {
      if(cluster.at<ushort>(y,x)==id) pad_partial.at<float>(y,x) = 1.0;
    }
  }
  GaussianBlur(pad, pad, Size(3,3), 0.0f);
  int sum = 0;
  for(int y=1; y<pad.rows - 1; y++)
  {
    for(int x=1; x<pad.cols - 1; x++)
    {
      float dx = pad.at<float>(y,x+1) - pad.at<float>(y,x-1);
      float dy = pad.at<float>(y+1, x) - pad.at<float>(y-1, x);
      float amp = sqrt(dx * dx + dy * dy);
      if(amp < 0.1) continue;
      float rad = atan2(dx, dy) * 180.0f / M_PI; // atan2(dy, dx)ではなく？
      int idx = cvRound(remap(rad, -180.0f, 180.0f, 0, FEATURE_HOG_BIN_NUM - 1));
      if(idx < 0 || idx >= FEATURE_HOG_BIN_NUM) continue;
      hog[idx]++;
      sum++;
    }
  }
  if(sum > 0)
  {
    for(int i=0; i<hog.size(); i++)
    {
      hog[i] /= sum;
    }
  }
}

void createReflectivityHistogram(const Mat &lidar_img, const Mat &cluster, int id, vector<float> &reflectivity)
{
  auto remap = [](float val, float from_low, float from_high, float to_low, float to_high)
  {
    return (val - from_low) * (to_high - to_low) / (from_high - from_low) + to_low;
  };
  reflectivity.resize(FEATURE_REF_BIN_NUM, .0f);
  int sum = 0;
  for(int y = 0; y < cluster.rows; y++)
  {
    for(int x = 0; x < cluster.cols; x++)
    {
      if(cluster.at<ushort>(y, x) == id)
      {
        float ref = lidar_img.at<Vec2f>(y, x)[0];
        int idx = cvRound(remap(ref, .0f, 1.0f, 0, FEATURE_REF_BIN_NUM - 1));
        if(idx < 0 || idx >= FEATURE_REF_BIN_NUM) continue;
        reflectivity[idx]++;
        sum++;
      }
    }
  }
  if(sum > 0)
  {
    for(int i = 0; i < reflectivity.size(); i++){
      reflectivity[i] /= sum;
    }
  } 
}

void extractObstacleFeatures(const TrackableObstacle &obstacle, vector<ObstacleFeature> &features)
{
  features.resize(obstacle.size);
  for(int i = 0; i<obstacle.size; i++)
  {
    auto &r = obstacle.rectangles[i];
    calculateHog(obstacle.clusters(r), i, features[i].hog);
    createReflectivityHistogram(obstacle.lidar_img(r), obstacle.clusters(r), i, features[i].reflectivity);
    features[i].width = obstacle.obstacles[i].width;
    features[i].height = obstacle.obstacles[i].height;
    features[i].center = obstacle.obstacles[i].getAbsoluteCenter();
  }
}

float matchHistogram(const vector<float> histo1, const vector<float> &histo2)
{
  /*** Histogram Intersectionによりヒストグラムの類似度を算出 ***/
  float score = .0f;
  int sz1 = histo1.size(), sz2 = histo2.size();
   	
  // cout << " histo1の要素数: " << sz1 << endl;
  // cout << " histo2の要素数: " << sz2 << endl;

  for(int i = 0; i < sz1; i++)
  {
    score = score + min(histo1[i],histo2[i]);
  }
  return score;
}

void matching(const vector<ObstacleFeature> &features1, const vector<ObstacleFeature> &features2, vector<int> &pairs)
{
  int sz1 = features1.size(), sz2 = features2.size();
  Mat score(sz1, sz2, CV_32FC1, Scalar(.0));
  /*** それぞれの障害物のスコアマトリックスを計算 ***/
  float hog_s, reflectivity_s, width_s, height_s; //スコアの各要素
  for(int i = 0; i < sz1; i++)
  {
    for(int j = 0; j < sz2; j++)
    {
      //4つの項目hog，reflectivity，width, height(それぞれ×0.25して総和の最大が1になるようにする)の総和をマトリックスに入れていく
      //hog,reflectivityは”matchHistogram関数”を呼び出して計算
      hog_s = 0.25 * matchHistogram(features1[i].hog, features2[j].hog);
      reflectivity_s = 0.25 * matchHistogram(features1[i].reflectivity, features2[j].reflectivity);
      //width, heightは比(小さい値/大きい値)で計算 
      if(features1[i].width >= features2[j].width){
        width_s = 0.25 * (features2[j].width / features1[i].width);
      } else {
        width_s = 0.25 * (features1[i].width / features2[j].width);
      }
      if(features1[i].height >= features2[j].height){
        height_s = 0.25 * (features2[j].height / features1[i].height);
      } else {
        height_s = 0.25 * (features1[i].height / features2[j].height);
      }
      auto &s = score.at<float>(i,j);
      s = hog_s + reflectivity_s + width_s + height_s;
    }
  }
  
  pairs = vector<int>(sz1, -1); //pairs[マトリックス縦] = (マトリックス横)
  /*** 障害物間で対応が取れているものを探索 ***/
  while(true){
    double max_val = .0;
    Point max_pt;
    for(int y = 0; y < score.rows; y++){
      for(int x = 0; x < score.cols; x++){
        auto &v = score.at<float>(y, x);
        if(v > max_val){
          max_val = v;
          max_pt.x = x;
          max_pt.y = y;
        }
      }
    }
    if(max_val < MATCH_THRESHOLD) break;
    if(max_pt.y < 0 || max_pt.y >= sz1) break;
    pairs[max_pt.y] = max_pt.x;
    for(int x = 0; x < sz2; x++) score.at<float>(max_pt.y, x) = .0f;
    for(int y = 0; y < sz1; y++) score.at<float>(y, max_pt.x) = .0f;
  }
}

void detect(const vector<LidarData> &points, TrackableObstacle &obstacle)
{
  vector<pair<ushort, ushort>> id_pairs;
  projectToImage(points, obstacle.lidar_img);
  int max_id = provisionalLabeling(obstacle.lidar_img, obstacle.clusters, id_pairs);
  max_id = integrateLabel(id_pairs, obstacle.clusters, max_id); // エラーの原因

  // cout << "max_id : " << max_id << endl;

  obstacle.size = relabel(obstacle.clusters, max_id); // == new_max_id

  // cout << "obstacle.size : " << obstacle.size << endl;

  extractRectangles(obstacle.clusters, obstacle.size, obstacle.rectangles);
  rectanglesToObstacle(obstacle, obstacle.obstacles);
  extractObstacleFeatures(obstacle, obstacle.features);
  obstacles_buffer.push_back(obstacle);
  points_buffer.push_back(points);
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
      if(range>MAX_RANGE || range<= .0f) continue; // 必須
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
      if(range>MAX_RANGE || range<= .0f) continue;
      int val = (int)remap(range, (float)MAX_RANGE, 0.0f, 30.0f, 255.0f);
      // int val = (int)remap(range, 0.0f, (float)MAX_RANGE, 30.0f, 255.0f);
      img.at<Vec3b>(y,x) = Vec3b(val, val, val);
    }
  }
}

void drawCorresponding(Mat &img1, Mat &img2, int &obs1_cnt)
{
  static const vector<Scalar> colors{
    Scalar(0, 0, 255), Scalar(0, 127, 255), Scalar(0, 255, 255), Scalar(0, 255, 127), 
    Scalar(0, 255, 0), Scalar(127, 255, 0), Scalar(255, 255, 0), Scalar(255, 0, 0), 
    Scalar(255, 0, 127), Scalar(255, 0, 255), Scalar(127, 0, 255)
  };
  if(obstacles_buffer.size() == 0) return;
  auto &obstacle1 = obstacles_buffer.back();
  auto &obstacle2 = obstacles_buffer[0];

  // drawObjects(obstacle1, img1);
  // drawObjects(obstacle2, img2);

  auto &pairs = obstacle1.pairs;
  int n = 0;

  obs1_cnt = obstacle1.rectangles.size() -1;
  cout << " cpp 61 now obstacle count = " << obstacle1.rectangles.size() -1 << endl;
  for(int i = 0; i < obstacle1.size; i++){
    if(pairs[i] >= obstacle2.size) continue;
    auto &r1 = obstacle1.rectangles, &r2 = obstacle2.rectangles;
    if(pairs[i] < 0){
      rectangle(img1, r1[i], Scalar(255, 255, 255), 2);
    } else{
      auto &flag = obstacle1.obstacles[i].flag;
      rectangle(img1, r1[i], colors[n % colors.size()], 2);
      rectangle(img2, r2[pairs[i]], colors[n % colors.size()], 2);
      n++;
    }
  }
}

void saveCroppedRectangle(const Mat &input_image, const Rect &roi,const string &directory_name, const string &file_name)
{
  Mat cropped_image = input_image(roi).clone();
  std::string output_path = directory_name + "/" + file_name;
  imwrite(output_path, cropped_image);
}

void rectangleReflect(const Mat &lidar_reflect_img, const Mat &camera_img)
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
    if(pixel_num<MIN_PIX_NUM || pixel_num>MAX_PIX_NUM || aspect_ratio<MIN_ASPECT_RATIO || aspect_ratio>MAX_ASPECT_RATIO)
    {
      continue;
    }
    cv::rectangle(lidar_reflect_img, cv::Rect(lefts[label], tops[label], widths[label], heights[label]), cv::Scalar(255,255,0), 0);
    // cout << "label : " << label << endl;
    int angle_h_pix = cvRound(((lidar_reflect_img.size().width)/2) - lefts[label]);
    int angle_v_pix = cvRound(((lidar_reflect_img.size().height)/2) - tops[label]);
    double angle_h = angle_h_pix * LIDAR_RESOLUTION_H;
    double angle_v = angle_v_pix * LIDAR_RESOLUTION_V;

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

    cv::rectangle(camera_img, cv::Rect(camera_left, camera_top, camera_width, camera_height), cv::Scalar(255,255,0), 2);
    // cv::Rect roi(lefts[label], tops[label], widths[label], heights[label]);
    // std::string directory_name = "/home/chiba/share/camera_lidar_data/tmp/000032";
    // std::string file_name = "cropped_image_" + std::to_string(label) + ".png";
    // saveCroppedRectangle(lidar_reflect_img, roi, directory_name, file_name);

    int pt1_x, pt1_y, pt2_x, pt2_y;
    if(camera_left - 3*camera_width < 0){
      pt1_x = 0;
    }else{
      pt1_x = camera_left - 3*camera_width;
    }

    if(camera_top - 0.5*camera_height < 0){
      pt1_y = 0;
    }else{
      pt1_y = camera_top - 0.5*camera_height;
    }

    if(camera_left + camera_width + 3*camera_width > camera_img.size().width){
      pt2_x = camera_img.size().width;
    }else{
      pt2_x = camera_left + camera_width + 3*camera_width;
    }

    if(camera_top + camera_height + 0.5*camera_height > camera_img.size().height * 0.5){
      pt2_y = camera_img.size().height * 0.5;
    }else{
      pt2_y = camera_top + camera_height + 0.5*camera_height;
    }
    cv::Point pt1(pt1_x, pt1_y);
    cv::Point pt2(pt2_x, pt2_y);
    cv::Rect rect(pt1, pt2);
    cv::rectangle(camera_img, rect, Scalar(0, 255, 0), 2);
  }
}

void saveRangeImage(const Mat &lidar_range_img)
{
  string directory_path = "/home/chiba/share/camera_lidar_data/tmp/000000";
  string file_path = "range_tmp.png";
  string output_path = directory_path + "/" + file_path;
  imwrite(output_path, lidar_range_img);
}

int main(int argc,char **argv){
  vector<LidarData> points;
  vector<fs::path> files_png;
  vector<fs::path> files_pcd;
  Mat lidar_img;
  Mat lidar_range_img;
  Mat lidar_reflect_img;

  double mgn = 2; // magnification : 倍率

  if(argc < 3){
    cout << "Usage is : " <<  argv[0] << "[image_directory_path][pcd_directory_path]" << endl;
    cout << "困ったら/home/chiba/share/camera_lidar_data/img/ /home/chiba/share/camera_lidar_data/pcd/を入力"<<endl;
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
    vector<LidarData> points;
    TrackableObstacle obstacle;
    Mat obstacle_now;
    Mat obstacle_past;

    /* カメラ画像の読み込み */
    Mat camera_img = imread(files_png[file_cnt].string(),1);
    cout << "camera file : " << files_png[file_cnt].string() << endl;
    /* 点群の読み込み */
    loadPCD(files_pcd[file_cnt].string(), points);
    detect(points, obstacle);

    int now_obs_cnt;

    // drawCorresponding(obstacle_now, obstacle_past, now_obs_cnt);
    now_obs_cnt_sum += now_obs_cnt;

    drawObjectsReflect(obstacle, lidar_reflect_img);
    drawObjectsRange(obstacle, lidar_range_img);

    rectangleReflect(lidar_reflect_img, camera_img);

    // saveRangeImage(lidar_range_img);

    // resize(obstacle_now, obstacle_now, Size(), mgn, mgn);
    // resize(obstacle_past, obstacle_past, Size(), mgn, mgn);

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