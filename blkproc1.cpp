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

  // 画像の読み込み
  cv::Mat rgb = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);
  if (rgb.empty()) {  // 画像ファイルが見つからない場合の処理
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
  cv::split(YCbCr, buf);        // planar形式に変換
  std::vector<cv::Mat> buf_f(3);

  // ------ ENCODE
  for (size_t c = 0; c < buf.size(); ++c) {
    buf[c].convertTo(buf_f[c], CV_32F);  // 浮動小数点に変換
    buf[c] -= 128;  // 全画素の値から128を引く（DCレベルシフト）
    blkproc(buf_f[c], blk::dct2);                      // 順方向のDCT
    blkproc(buf_f[c], blk::quantize, qtables[c > 0]);  // 量子化
    // ハフマン符号化
  }
  // ------ DECODE
  for (size_t c = 0; c < buf.size(); ++c) {
    // ハフマン復号
    blkproc(buf_f[c], blk::dequantize, qtables[c > 0]);  // 逆量子化
    blkproc(buf_f[c], blk::idct2);                       // 逆方向のDCT
    buf[c] += 128;
    buf_f[c].convertTo(buf[c], CV_8U);  // 8bitの符号なし整数に変換
  }

  cv::merge(buf, YCbCr);  // inerleave形式に変換
  myYCbCr2BGR(YCbCr, recimg);

  // 画像の表示
  cv::imshow("Original", rgb);
  cv::imshow("Reconstructed", recimg);
  // キー入力を待つ'q'で終了
  int keycode;
  do {
    keycode = cv::waitKey();
  } while (keycode != 'q');

  // 全てのウィンドウを破棄
  cv::destroyAllWindows();

  return EXIT_SUCCESS;
}
