#include <iostream>

#include <argparse/argparse.hpp>

#include <vkgs/engine/engine.h>

int main(int argc, char** argv) {
  argparse::ArgumentParser parser("vkgs");
  parser.add_argument("-i", "--input").help("input ply file.");
  try {
    parser.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  try {
    vkgs::Engine engine;

    std::string plyFilepath;
    if (parser.is_used("input")) {
      plyFilepath = parser.get<std::string>("input");
    }

    engine.Run(plyFilepath);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
