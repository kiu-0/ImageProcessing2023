#include <cstdio>
#include <cstdlib>
#include <opencv2/opencv.hpp>
#include <vector>

#include "qtables.hpp"
#include "tools.hpp"

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Input image is required.\n");
    return EXIT_FAILURE;
  }
  // if (argc < 3) {
  //   printf("Qfactor is missing.\n");
  //   return EXIT_FAILURE;
  //}

  // �摜�̓ǂݍ���
  cv::Mat rgb = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);
  if (rgb.empty()) {  // �摜�t�@�C����������Ȃ��ꍇ�̏���
    printf("Input image is not found.\n");
    return EXIT_FAILURE;
  }

  int QF = 5;  // strtol(argv[2], nullptr, 10);
  // if (QF < 0 || QF > 100) {
  //  printf("The valid range of qfactor is from 0 to 100.\n");
  // return EXIT_FAILURE;
  //}

  int qtables[2][DCTSIZE * DCTSIZE];
  create_qtable(0, QF, &qtables[0][0]);
  create_qtable(1, QF, &qtables[1][0]);

  cv::Mat YCbCr = rgb.clone();
  cv::Mat recimg = rgb.clone();

  myBGR2YCbCr(rgb, YCbCr);
  std::vector<cv::Mat> buf(3);  // Y: buf[0], Cb: buf[1], Cr: Buf[2]
  cv::split(YCbCr, buf);        // planar�`���ɕϊ�
  std::vector<cv::Mat> buf_f(3);

  // ------ ENCODE
  for (size_t c = 0; c < buf.size(); ++c) {
    buf[c].convertTo(buf_f[c], CV_32F);  // ���������_�ɕϊ�
    buf[c] -= 128;  // �S��f�̒l����128�������iDC���x���V�t�g�j
    blkproc(buf_f[c], blk::dct2);                      // ��������DCT
    blkproc(buf_f[c], blk::quantize, qtables[c > 0]);  // �ʎq��
    // �n�t�}��������
  }
  // ------ DECODE
  for (size_t c = 0; c < buf.size(); ++c) {
    // �n�t�}������
    blkproc(buf_f[c], blk::dequantize, qtables[c > 0]);  // �t�ʎq��
    blkproc(buf_f[c], blk::idct2);                       // �t������DCT
    buf[c] += 128;
    buf_f[c].convertTo(buf[c], CV_8U);  // 8bit�̕����Ȃ������ɕϊ�
  }

  cv::merge(buf, YCbCr);  // inerleave�`���ɕϊ�
  myYCbCr2BGR(YCbCr, recimg);

  // �摜�̕\��
  cv::imshow("Original", rgb);
  cv::imshow("Reconstructed", recimg);
  // �L�[���͂�҂�'q'�ŏI��
  int keycode;
  do {
    keycode = cv::waitKey();
  } while (keycode != 'q');

  // �S�ẴE�B���h�E��j��
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
