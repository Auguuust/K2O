//
//  file_generator.h
//  YCSB-cpp
//
//  Load keys from file for real-world datasets (Facebook, Wiki, etc.)
//

#ifndef YCSB_C_FILE_GENERATOR_H_
#define YCSB_C_FILE_GENERATOR_H_

#include "generator.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstdint>

namespace ycsbc {

class FileGenerator : public Generator<uint64_t> {
 public:
  FileGenerator(const std::string& filename) : current_index_(0), last_value_(0) {
    LoadKeysFromFile(filename);
  }

  uint64_t Next() {
    if (keys_.empty()) {
      std::cerr << "Error: No keys loaded from file!" << std::endl;
      return 0;
    }
    if (current_index_ >= keys_.size()) {
      current_index_ = 0;  // Wrap around if we reach the end
    }
    last_value_ = keys_[current_index_++];
    return last_value_;
  }

  uint64_t Last() {
    return last_value_;
  }

  size_t GetKeyCount() const {
    return keys_.size();
  }

  // Get key by index (for random access patterns)
  uint64_t GetKey(size_t index) {
    if (index >= keys_.size()) {
      return keys_[index % keys_.size()];
    }
    return keys_[index];
  }

 private:
  std::vector<uint64_t> keys_;
  size_t current_index_;
  uint64_t last_value_;

  void LoadKeysFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
      std::cerr << "Error: Cannot open file: " << filename << std::endl;
      return;
    }

    std::cout << "Loading keys from file: " << filename << std::endl;
    std::string line;
    size_t line_count = 0;
    
    while (std::getline(file, line)) {
      line_count++;
      if (line.empty()) continue;

      // Extract the numeric part (handle both "INSERT 0000..." and "0000..." formats)
      size_t pos = line.find_last_of(" \t");
      std::string key_str;
      
      if (pos != std::string::npos) {
        // Format: "INSERT 0000..."
        key_str = line.substr(pos + 1);
      } else {
        // Format: "0000..."
        key_str = line;
      }

      // Convert to uint64_t
      try {
        uint64_t key = std::stoull(key_str);
        keys_.push_back(key);
      } catch (const std::exception& e) {
        std::cerr << "Warning: Failed to parse line " << line_count 
                  << ": " << line << std::endl;
        continue;
      }

      // Progress indicator for large files
      if (line_count % 1000000 == 0) {
        std::cout << "Loaded " << line_count << " keys..." << std::endl;
      }
    }

    file.close();
    std::cout << "Successfully loaded " << keys_.size() << " keys from " 
              << filename << std::endl;
  }
};

} // namespace ycsbc

#endif // YCSB_C_FILE_GENERATOR_H_
