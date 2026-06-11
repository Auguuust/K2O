//  Copyright (c) 2011-present, Facebook, Inc.  All rights reserved.
//  This source code is licensed under both the GPLv2 (found in the
//  COPYING file in the root directory) and Apache 2.0 License
//  (found in the LICENSE.Apache file in the root directory).

#include "table/block_based/learned_index.h"
#include "test_util/testharness.h"
#include <vector>
#include <string>

namespace ROCKSDB_NAMESPACE {

class LearnedIndexTest : public testing::Test {
 public:
  LearnedIndexTest() = default;
};

// Test KeyEncoder with simple keys
TEST_F(LearnedIndexTest, KeyEncoderBasic) {
  std::vector<std::string> keys = {"a", "b", "c"};
  
  KeyEncoder encoder;
  encoder.AnalyzeKeys(keys);
  
  // Encoded values should be monotonically increasing
  double enc_a = encoder.EncodeKey("a");
  double enc_b = encoder.EncodeKey("b");
  double enc_c = encoder.EncodeKey("c");
  
  ASSERT_LT(enc_a, enc_b);
  ASSERT_LT(enc_b, enc_c);
}

// Test KeyEncoder with string keys
TEST_F(LearnedIndexTest, KeyEncoderStrings) {
  std::vector<std::string> keys = {"aab", "bdd", "bcb"};
  
  KeyEncoder encoder;
  encoder.AnalyzeKeys(keys);
  
  double enc1 = encoder.EncodeKey("aab");
  double enc2 = encoder.EncodeKey("bdd");
  double enc3 = encoder.EncodeKey("bcb");
  
  // Should maintain order
  ASSERT_LT(enc1, enc2);
  ASSERT_LT(enc2, enc3);
}

// Test KeyEncoder serialization
TEST_F(LearnedIndexTest, KeyEncoderSerialization) {
  std::vector<std::string> keys = {"abc", "def", "ghi"};
  
  KeyEncoder encoder1;
  encoder1.AnalyzeKeys(keys);
  
  // Serialize
  std::string serialized;
  encoder1.SerializeMetadata(&serialized);
  
  // Deserialize
  KeyEncoder encoder2;
  ASSERT_TRUE(encoder2.DeserializeMetadata(serialized));
  
  // Both should produce same encodings
  ASSERT_EQ(encoder1.EncodeKey("abc"), encoder2.EncodeKey("abc"));
  ASSERT_EQ(encoder1.EncodeKey("def"), encoder2.EncodeKey("def"));
}

// Test LinearModel
TEST_F(LearnedIndexTest, LinearModelBasic) {
  LinearModel model(2.0, 10.0);  // y = 2x + 10
  
  ASSERT_EQ(model.Predict(0), 10.0);
  ASSERT_EQ(model.Predict(5), 20.0);
  ASSERT_EQ(model.Predict(10), 30.0);
}

// Test LearnedIndex training with simple data
TEST_F(LearnedIndexTest, LearnedIndexTraining) {
  std::vector<TrainingEntry> training_data;
  
  // Simulate cumulative bytes
  training_data.emplace_back("key000", 100);
  training_data.emplace_back("key001", 200);
  training_data.emplace_back("key002", 300);
  training_data.emplace_back("key003", 400);
  training_data.emplace_back("key004", 500);
  
  LearnedIndex index;
  uint64_t target_block_size = 100;
  
  ASSERT_TRUE(index.Train(training_data, target_block_size));
  ASSERT_TRUE(index.IsTrained());
  
  // Predictions should match blocks
  ASSERT_EQ(index.PredictBlock("key000"), 0);
  ASSERT_EQ(index.PredictBlock("key001"), 1);
  ASSERT_EQ(index.PredictBlock("key002"), 2);
}

// Test LearnedIndex serialization
TEST_F(LearnedIndexTest, LearnedIndexSerialization) {
  std::vector<TrainingEntry> training_data;
  training_data.emplace_back("key001", 100);
  training_data.emplace_back("key002", 200);
  training_data.emplace_back("key003", 300);
  
  LearnedIndex index1;
  ASSERT_TRUE(index1.Train(training_data, 100));
  
  // Serialize
  std::string serialized;
  index1.Serialize(&serialized);
  
  // Deserialize
  LearnedIndex index2;
  ASSERT_TRUE(index2.Deserialize(serialized));
  ASSERT_TRUE(index2.IsTrained());
  
  // Should produce same predictions
  ASSERT_EQ(index1.PredictBlock("key001"), index2.PredictBlock("key001"));
  ASSERT_EQ(index1.PredictBlock("key002"), index2.PredictBlock("key002"));
}

// Test BlockLocationMapper
TEST_F(LearnedIndexTest, BlockLocationMapper) {
  BlockLocationMapper mapper;
  
  mapper.AddBlock(0, 100);
  mapper.AddBlock(100, 150);
  mapper.AddBlock(250, 200);
  
  ASSERT_EQ(mapper.GetNumBlocks(), 3);
  
  auto loc0 = mapper.GetBlockLocation(0);
  ASSERT_EQ(loc0.offset, 0);
  ASSERT_EQ(loc0.size, 100);
  
  auto loc1 = mapper.GetBlockLocation(1);
  ASSERT_EQ(loc1.offset, 100);
  ASSERT_EQ(loc1.size, 150);
  
  auto loc2 = mapper.GetBlockLocation(2);
  ASSERT_EQ(loc2.offset, 250);
  ASSERT_EQ(loc2.size, 200);
}

// Test BlockLocationMapper serialization
TEST_F(LearnedIndexTest, BlockLocationMapperSerialization) {
  BlockLocationMapper mapper1;
  mapper1.AddBlock(0, 4096);
  mapper1.AddBlock(4096, 4096);
  mapper1.AddBlock(8192, 4096);
  
  // Serialize
  std::string serialized;
  mapper1.Serialize(&serialized);
  
  // Deserialize
  BlockLocationMapper mapper2;
  ASSERT_TRUE(mapper2.Deserialize(serialized));
  
  ASSERT_EQ(mapper1.GetNumBlocks(), mapper2.GetNumBlocks());
  
  for (uint32_t i = 0; i < mapper1.GetNumBlocks(); ++i) {
    auto loc1 = mapper1.GetBlockLocation(i);
    auto loc2 = mapper2.GetBlockLocation(i);
    ASSERT_EQ(loc1.offset, loc2.offset);
    ASSERT_EQ(loc1.size, loc2.size);
  }
}

// Test with realistic key distribution
TEST_F(LearnedIndexTest, RealisticKeyDistribution) {
  std::vector<TrainingEntry> training_data;
  
  // Simulate user keys with realistic distribution
  uint64_t cumulative = 0;
  for (int i = 0; i < 1000; ++i) {
    char key_buf[32];
    snprintf(key_buf, sizeof(key_buf), "user%08d", i * 100);
    cumulative += 100;  // Assume 100 bytes per entry
    training_data.emplace_back(key_buf, cumulative);
  }
  
  LearnedIndex index;
  ASSERT_TRUE(index.Train(training_data, 4096));
  
  // Check monotonicity of predictions
  uint32_t prev_block = 0;
  for (int i = 0; i < 1000; ++i) {
    char key_buf[32];
    snprintf(key_buf, sizeof(key_buf), "user%08d", i * 100);
    uint32_t block = index.PredictBlock(key_buf);
    ASSERT_GE(block, prev_block);  // Should be non-decreasing
    prev_block = block;
  }
  
  auto stats = index.GetStats();
  std::cout << "Model stats: slope=" << stats.model_slope 
            << " intercept=" << stats.model_intercept << std::endl;
}

// Test edge cases
TEST_F(LearnedIndexTest, EdgeCases) {
  // Empty training data
  {
    LearnedIndex index;
    std::vector<TrainingEntry> empty;
    ASSERT_FALSE(index.Train(empty, 4096));
  }
  
  // Single entry
  {
    LearnedIndex index;
    std::vector<TrainingEntry> single;
    single.emplace_back("key", 100);
    ASSERT_FALSE(index.Train(single, 4096));  // Need at least 2 points
  }
  
  // Two entries (minimum)
  {
    LearnedIndex index;
    std::vector<TrainingEntry> two;
    two.emplace_back("key1", 100);
    two.emplace_back("key2", 200);
    ASSERT_TRUE(index.Train(two, 100));
  }
}

}  // namespace ROCKSDB_NAMESPACE

int main(int argc, char** argv) {
  ROCKSDB_NAMESPACE::port::InstallStackTraceHandler();
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
