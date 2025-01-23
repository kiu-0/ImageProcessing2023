#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <opencv2/opencv.hpp>
#include <vector>

constexpr int DCTSIZE = 8;

int clip(int v);
void myBGR2YCbCr(cv::Mat &in, cv::Mat &out);
void myYCbCr2BGR(cv::Mat &in, cv::Mat &out);
void quantize(cv::Mat &in, float stepsize);
void blkproc(cv::Mat &in, std::function<void(cv::Mat &)> func);
void mozaic(cv::Mat &);
void line_mozaic(cv::Mat &);

void block_dct2(cv::Mat &);
void block_idct2(cv::Mat &);

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Input image is required.\n");
    return EXIT_FAILURE;
  }

  // 画像の読み込み
  cv::Mat rgb = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);
  if (rgb.empty()) {  // 画像ファイルが見つからない場合の処理
    printf("Input image is not found.\n");
    return EXIT_FAILURE;
  }

  // int W = rgb.cols;
  // int H = rgb.rows;
  // int nc = rgb.channels();

  cv::Mat YCbCr = rgb.clone();
  cv::Mat recimg = rgb.clone();

  myBGR2YCbCr(rgb, YCbCr);
  std::vector<cv::Mat> buf(3);  // Y: buf[0], Cb: buf[1], Cr: Buf[2]
  cv::split(YCbCr, recimg);     // planar形式に変換
  // cv::imshow("Y only", buf[0]);

  // for (size_t c = 0; c < buf.size(); ++c) {
  //  blkproc(buf[c], line_mozaic);
  //}

  std::vector<cv::Mat> buf_f(3);
  //------ENCODE
  for (size_t c = 0; c < buf.size(); ++c) {
    buf[c].convertTo(buf_f[c], CV_32F);  // 浮動小数点に変換
    buf[c] -= 128;  // 全画素の値から１２８を引く（DCレベルシフト）
    blkproc(buf_f[c], block_dct2);  // 順方向のDCT
    // 量子化
    // ハフマン符号化
  }
  //------DECODE
  for (size_t c = 0; c < buf.size(); ++c) {
    // ハフマン複号
    // 逆量子化
    blkproc(buf_f[c], block_idct2);  // 逆方向のDCT
    buf[c] += 128;
    buf_f[c].convertTo(buf[c], CV_8U);  // 8bitの符号なし整数に変換
  }

  cv::merge(buf, YCbCr);  // interleave形式に変換
  myYCbCr2BGR(YCbCr, recimg);

  // 画像の表示
  cv::imshow("Originai", rgb);
  // cv::imshow("RGB", rgb);
  // cv::imshow("YCbCr->RGB", recimg);
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

int clip(int v) {
  if (v > 255) {
    v = 255;
  }
  if (v < 0) {
    v = 0;
  }
  return v;
}

void myBGR2YCbCr(cv::Mat &in, cv::Mat &out) {
  int W = in.cols;
  int H = in.rows;
  int nc = in.channels();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < nc * W; x += nc) {
      // BGR|BGR|BGR|BGR|...
      int idx = y * nc * W + x;
      int B = in.data[idx];
      int G = in.data[idx + 1];
      int R = in.data[idx + 2];
      int Y = 0.299 * R + 0.587 * G + 0.114 * B;
      int Cb = -0.169 * R - 0.331 * G + 0.5 * B;
      int Cr = 0.5 * R - 0.419 * G - 0.081 * B;
      out.data[idx] = clip(Y);
      out.data[idx + 1] = clip(Cb + 128);
      out.data[idx + 2] = clip(Cr + 128);
    }
  }
}

void myYCbCr2BGR(cv::Mat &in, cv::Mat &out) {
  int W = in.cols;
  int H = in.rows;
  int nc = in.channels();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < nc * W; x += nc) {
      // BGR|BGR|BGR|BGR|...
      int idx = y * nc * W + x;
      int Y = in.data[idx];
      int Cb = in.data[idx + 1] - 128;
      int Cr = in.data[idx + 2] - 128;

      int R = Y + 1.402 * Cr;
      int G = Y - 0.344 * Cb - 0.714 * Cr;
      int B = Y + 1.772 * Cb;
      out.data[idx] = clip(B);
      out.data[idx + 1] = clip(G);
      out.data[idx + 2] = clip(R);
    }
  }
}

void quantize(cv::Mat &in, float stepsize) {
  int W = in.cols;
  int H = in.rows;
  int nc = in.channels();

  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < nc * W; x += nc) {
      // BGR|BGR|BGR|BGR|...
      int idx = y * nc * W + x;
      int Y = in.data[idx];
      int Cb = in.data[idx + 1];
      int Cr = in.data[idx + 2];

      // Y = floor((floor(Y / stepsize) + 0.5) * stepsize);
      Cb = floor((floor(Cb / stepsize) + 0.5) * stepsize);
      Cr = floor((floor(Cr / stepsize) + 0.5) * stepsize);

      in.data[idx] = clip(Y);
      in.data[idx + 1] = clip(Cb);
      in.data[idx + 2] = clip(Cr);
    }
  }
}

void blkproc(cv::Mat &in, std::function<void(cv::Mat &)> func) {
  for (int y = 0; y < in.rows; y += DCTSIZE) {
    for (int x = 0; x < in.cols; x += DCTSIZE) {
      cv::Mat blk_in, blk_out;
      blk_in = in(cv::Rect(x, y, DCTSIZE, DCTSIZE)).clone();
      blk_out = in(cv::Rect(x, y, DCTSIZE, DCTSIZE));
      func(blk_in);
      blk_in.convertTo(blk_out, blk_out.type());
    }
  }
}

void mozaic(cv::Mat &in) {  // 8×8　ブロック内の値を左上の値で塗りつぶす処理
  // //8×8　ブロック内の各行を各行の先頭の値で塗りつぶす処理
  if (in.rows != DCTSIZE || in.cols != DCTSIZE || in.channels() != 1) {
    printf("input for mozaic() shall be 8×8 and monochrome.\n");
    exit(EXIT_FAILURE);
  }
  for (int y = 0; y < in.rows; ++y) {
    for (int x = 0; x < in.cols; ++x) {
      // in.data[y * DCTSIZE + x] = in.data[0];
      in.data[y * DCTSIZE + x] = in.data[0];  //[y * DCTSIZE];
    }
  }
}

void line_mozaic(
    cv::Mat &in) {  // 8×8　ブロック内の値を左上の値で塗りつぶす処理
  // //8×8　ブロック内の各行を各行の先頭の値で塗りつぶす処理
  if (in.rows != DCTSIZE || in.cols != DCTSIZE || in.channels() != 1) {
    printf("input for mozaic() shall be 8×8 and monochrome.\n");
    exit(EXIT_FAILURE);
  }
  for (int y = 0; y < in.rows; ++y) {
    for (int x = 0; x < in.cols; ++x) {
      // in.data[y * DCTSIZE + x] = in.data[0];
      in.data[y * DCTSIZE + x] = in.data[y * DCTSIZE];
    }
  }
}

void block_dct2(cv::Mat &in) {
  if (in.rows != DCTSIZE || in.cols != DCTSIZE || in.channels() != 1) {
    printf("input for mozaic() shall be 8×8 and monochrome.\n");
    exit(EXIT_FAILURE);
  }
  cv::dct(in, in);
}

void block_idct2(cv::Mat &in) {
  if (in.rows != DCTSIZE || in.cols != DCTSIZE || in.channels() != 1) {
    printf("input for mozaic() shall be 8×8 and monochrome.\n");
    exit(EXIT_FAILURE);
  }
  cv::idct(in, in);
}
