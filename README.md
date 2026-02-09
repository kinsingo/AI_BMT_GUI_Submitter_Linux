> **Last Updated:** 2026-02-09 (Version 2.2)

## Environment
1.  ISA(Instruction Set Architecture) : AMD64(x86_64)
2.  OS : Ubuntu 22.04 LTS, Ubuntu 24.04 LTS

## Submitter User Guide Steps
Step1) Build System Set-up  
Step2) Interface Implementation  
Step3) Build and Start BMT

## Step 1) Build System Set-up (Installation Guide for Ubuntu)
**1. Install Packages**
- Open a terminal and run the following commands to install CMake, g++ compiler, Ninja Build System, and EGL Library.
  ```bash
  sudo apt update
  sudo apt install cmake                     # CMake
  sudo apt install build-essential           # GCC, G++, Make
  sudo apt-get install ninja-build           # Ninja
  sudo apt install libgl1 libgl1-mesa-dev libxcb-cursor0 libxkbcommon-x11-0 libxcb-xinerama0  # EGL and OpenGL
  sudo apt install unzip                     # unzip
  ```

**2. Verify the Installation**

- You can check the versions of the installed tools by running the following commands. If these commands return version information for each tool, the installation was successful.
  ```bash
  cmake --version
  gcc --version
  ninja --version
  dpkg -l | grep -E 'libgl1|libgl1-mesa-dev'
  ```

## Step2) Interface Implementation
1. Implement `AI_BMT_Interface` to operate with the intended AI Processing Unit (e.g., CPU, GPU, NPU).

```cpp
#ifndef AI_BMT_INTERFACE_H
#define AI_BMT_INTERFACE_H
#include "label_type.h"
using namespace std;

class EXPORT_SYMBOL AI_BMT_Interface
{
public:
   virtual ~AI_BMT_Interface(){}

    // Optional: override to provide system metadata.
    // Returned values will be stored in the database (used for benchmarking context).
   virtual Optional_Data getOptionalData();

   // return the implemented interface task type. 
   virtual InterfaceType getInterfaceType() = 0;

   // This initialize(..) function is guaranteed to be called before preprocess(..) and infer(..) are executed.
   // The submitter can load the model using the provided modelPath
   virtual void initialize(string modelPath) = 0;

   // Vision tasks: preprocessing & inference
   // - preprocessVisionData: convert raw image file into model input format
   // - inferVision: run inference on preprocessed data and return results
   virtual VariantType preprocessVisionData(const string& imagePath) {throw runtime_error("preprocessVisionData(..) should be implemented for vision task");}
   virtual vector<BMTVisionResult> inferVision(const vector<VariantType>& data) {throw runtime_error("inferVision(..) should be implemented for vision task");}

   // LLM tasks: preprocessing & inference
   // - preprocessLLMData: convert raw text input into model input format
   // - inferLLM: run inference on preprocessed data and return results
   virtual VariantType preprocessLLMData(const LLMPreprocessedInput& llmData) {throw runtime_error("LLMPreprocessedInput(..) should be implemented for llm task");}
   virtual vector<BMTLLMResult> inferLLM(const vector<VariantType>& data) {throw runtime_error("inferLLM(..) should be implemented for llm task");}

   // LLM MMLU tasks: first token generation for TTFT measurement
   // - inferFirstToken: generate only the first token (AI-BMT will measure the time internally)
   // - Returns void (we only measure TTFT, don't care about the actual first token output)
   // - Only used for MMLU tasks that require TTFT measurement
   virtual void inferFirstToken(const VariantType& data) {throw runtime_error("inferFirstToken(..) should be implemented for MMLU task");}
};

#endif // AI_BMT_INTERFACE_H
```

## Step3) Build and Start BMT

**1. Generate the Ninja build system using cmake**

- Run the following command to remove existing cache
  ```bash
  rm -rf CMakeCache.txt CMakeFiles .ninja* build.ninja rules.ninja \
       cmake_install.cmake compile_commands.json qtcsettings.cmake .qtc AI_BMT_GUI_Submitter
  ```
- Run the following command to execute CMake in the current directory (usually the build directory). This command will generate the Ninja build system based on the CMakeLists.txt file located in the parent directory. Once successfully executed, the project will be ready to be built using Ninja.
  ```bash
  cmake -G "Ninja" ..
  ```

**2. Setting Library Path for Executable in Current Directory**

- Run the following command to make the executable(AI_BMT_GUI_Submitter) can reference the libraries located in the lib folder of the current directory.
  ```bash
  export LD_LIBRARY_PATH=$(pwd)/lib
  ```

**3. Build the project**

- Run the following command to build the project using the build system configured by CMake in the current directory. This will compile the project and create the executable AI_BMT_GUI_Submitter.exe in the build folder.
  ```bash
  cmake --build .
  ```

**4. Start Performance Analysis**

- Run the following command to start created excutable. When the GUI Popup, Click [Start BMT] button to start AI Performance Analysis.
  ```bash
  ./AI_BMT_GUI_Submitter
  ```

**Run all commands at once (For Initial Build)**

```bash
sudo apt update
sudo apt install cmake
sudo apt install build-essential
sudo apt-get install ninja-build
sudo apt-get install libgl1 libgl1-mesa-dev
sudo apt install unzip
rm -rf CMakeCache.txt CMakeFiles .ninja* build.ninja rules.ninja \
       cmake_install.cmake compile_commands.json qtcsettings.cmake .qtc AI_BMT_GUI_Submitter
cmake -G "Ninja" ..
export LD_LIBRARY_PATH=$(pwd)/lib
cmake --build .
./AI_BMT_GUI_Submitter
```

**Run all commands at once (For Rebuild)**

- Using following commands in `build/` directory.
```bash
rm -rf CMakeCache.txt CMakeFiles .ninja* build.ninja rules.ninja \
       cmake_install.cmake compile_commands.json qtcsettings.cmake .qtc AI_BMT_GUI_Submitter
cmake -G "Ninja" ..
export LD_LIBRARY_PATH=$(pwd)/lib
cmake --build .
./AI_BMT_GUI_Submitter
```

**Execute AI-BMT App**

- Using following commands in `build/` directory.
```bash
export LD_LIBRARY_PATH=$(pwd)/lib
./AI_BMT_GUI_Submitter
```
