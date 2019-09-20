// Copyright 2019 Intel Corporation.

#pragma once

#include <fstream>
#include <functional>
#include <string>

#include <boost/filesystem.hpp>

namespace vertexai {

std::string ReadFile(const boost::filesystem::path& path, bool binary = false);

void WriteFile(const boost::filesystem::path& path,  //
               const std::string& contents,          //
               bool binary = false,                  //
               bool append = false);

void WriteFile(const boost::filesystem::path& path,  //
               bool binary,                          //
               bool append,                          //
               const std::function<void(std::ofstream& fout)>& writer);

}  // namespace vertexai
