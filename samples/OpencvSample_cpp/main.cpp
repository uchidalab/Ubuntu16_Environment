#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "**Usage***********************" << std::endl;
    std::cout << " argv[1]: (fs::path) image filepath" << std::endl << std::endl;
    return -1;
  }

  cv::Mat img = cv::imread(argv[1]);
  if (img.empty()) {
    std::cerr << "Failed to read image from \"" << argv[1] << "\"" << std::endl << std::endl;
    exit(-1);
  }
  else {
    std::cout << "Success to read image from \"" << argv[1] << "\"" << std::endl;
    std::cout << "size: " << img.size() << std::endl << std::endl;
  }

  // cv::namedWindow("image", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
  // cv::imshow("", img);
  // cv::waitKey();

  return 0;
}