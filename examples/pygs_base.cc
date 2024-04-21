#include <iostream>

#include <argparse/argparse.hpp>

#include <pygs/engine/engine.h>

int main(int argc, char** argv) {
  argparse::ArgumentParser parser("pygs");
  parser.add_argument("-i", "--input").help("input ply file.");
  try {
    parser.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << parser;
    return 1;
  }

  try {
    pygs::Engine engine;

    if (parser.is_used("input")) {
      auto ply_filepath = parser.get<std::string>("input");
      engine.LoadSplats(ply_filepath);
    }

    engine.Run();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
  }

  return 0;
}
