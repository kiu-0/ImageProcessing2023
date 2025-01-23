#include <cstdio>
#include <cstdlib>
#include <memory>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Input image is required.\n");
    return EXIT_FAILURE;
  }

  // 画像の読み込み
  cv::Mat img = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);

  if (img.empty()) {  // 画像ファイルが見つからない場合の処理
    printf("Input image is not found.\n");
    return EXIT_FAILURE;
  }

  cv::Mat dst;
  int type = cv::BORDER_REFLECT_101;

  const int xsize = 3;
  const int ysize = 3;
  float c[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};  // メモリの確保
  auto coeff = std::make_unique<float[]>(xsize * ysize);
  for (int i = 0; i < xsize * ysize; ++i) {
    coeff[i] = c[i];
  };
  cv::Mat filter(cv::Size(xsize, ysize), CV_32FC1, coeff.get());

  cv::filter2D(img, dst, CV_8U, filter, cv::Point(-1, -1), 0.0, type);

  // 画像の表示
  cv::imshow("original", img);
  cv::imshow("extended", dst);

  // キー入力を待つ
  cv::waitKey();
  // 全てのウィンドウを破棄
  cv::destroyAllWindows();

  // delete[] coeff;  // メモリの開放、unique_ptrの場合不要

  return EXIT_SUCCESS;
}
