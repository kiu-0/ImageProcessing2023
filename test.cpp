#include <cmath>
#include <cstdint>
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

  // for (int y = 0; y < 1; y++) {
  // for (int x = 0; x < 4; x++) {
  // int idx = y * img.cols + x;
  //   printf("%3d", img.data[idx]);
  //}
  //}
  // printf("\n");

  auto clip = [](int v) {
    // 0-255�͈̔͂ɉ�f�l�����߂邽�߂̃N���b�s���O�������s�������_��
    if (v > 255) {
      v = 255;
    }
    if (v < 0) {
      v = 0;
    }
    return v;
  };

  auto gamma_correction = [](int v, double gamma) {
    // �K���}�␳���s�������_��
    double in = static_cast<double>(v) / 255.0;
    double out = pow(in, gamma) * 255.0;
    return static_cast<int>(out);
  };

  auto quantization = [](int v, double delta) {
    double x = static_cast<double>(v);
    double Qx = floor(x / delta + 0.5);
    return Qx;
  };

  auto dequantization = [](int v, double delta) {
    double rec = v * delta;
    return static_cast<int>(rec);
  };

  const int W = img.cols;
  const int H = img.rows;
  uint8_t *pixels = img.data;
  const double delta = 30.0;
  // Stride access
  for (int Y = 0; Y < H; ++Y) {
    for (int X = 0; X < W; ++X) {
      // �X�g���C�g�A�N�Z�X�ɂ�郁������̃A�h���X�̌v�Z
      int idx = Y * W + X;
      // ��f�l�̎擾
      int val = pixels[idx];
      // ����
      val = dequantization(quantization(val, delta), delta);

      // �������ʂ̉�f�l���������ɏ�������
      pixels[idx] = clip(val);
    }
  }
  // printf("\n");

  // �摜�̕\��
  cv::imshow("window", img);
  // �L�[���͂�҂�
  cv::waitKey();
  // �S�ẴE�B���h�E��j��
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
