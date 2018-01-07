#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

namespace fs = boost::filesystem;

fs::path convertChars2Path(const char* cs){
  std::string str_path(cs);
  fs::path path = str_path;
  return path;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cout << "**Usage***********************" << std::endl;
    std::cout << " argv[1]: (fs::path) path of directory" << std::endl << std::endl;
    return -1;
  }

  fs::path dirpath = convertChars2Path(argv[1]);
  BOOST_FOREACH(
      const fs::path& path,
      std::make_pair(fs::directory_iterator(dirpath), fs::directory_iterator())
  )
  {
    std::cout << path.string() << std::endl;
  }
  std::cout << std::endl;

  return 0;
}