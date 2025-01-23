#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Input image is required.\n");
    return EXIT_FAILURE;
  }

  // �摜�̓ǂݍ���
  cv::Mat img = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);

  if (img.empty()) {  // �摜�t�@�C����������Ȃ��ꍇ�̏���
    printf("Input image is not found.\n");
    return EXIT_FAILURE;
  }

  cv::Mat dst;
  int top, left, bottom, right;
  top = img.rows;
  bottom = img.rows;
  left = img.cols;
  right = img.cols;
  int type = cv::BORDER_REPLICATE;
  cv::copyMakeBorder(img, dst, top, bottom, left, right, type);

  // �摜�̕\��
  cv::imshow("original", img);
  cv::imshow("extended", dst);

  // �L�[���͂�҂�
  cv::waitKey();
  // �S�ẴE�B���h�E��j��
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
