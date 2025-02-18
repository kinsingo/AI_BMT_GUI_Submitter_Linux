> **Last Updated:** 2025-02-10
## Environment
 1. ISA(Instruction Set Architecture) : AMD64(x86_64)
 2. OS : Ubuntu 22.04, 24.04

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
   sudo apt install libgl1 libgl1-mesa-dev    # EGL and OpenGL
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
- Implement the overridden functions in the `Virtual_Submitter_Implementation` class within `main.cpp`  
: Ensure that these functions are implemented to operate correctly in the intended calculator (e.g., CPU, GPU, NPU).

## Step3) Build and Start BMT
**1. Open the Linux terminal**  
  - Clone and navigate to the build directory using the following command
    ```bash
    git clone https://github.com/kinsingo/SNU_BMT_GUI_Submitter_Linux.git
    cd SNU_BMT_GUI_Submitter_Windows/build
    ```
  
**2. Generate the Ninja build system using cmake**  
  - Run the following command to remove existing cache  
    ```bash
    rm -rf CMakeCache.txt CMakeFiles
    ```
  - Run the following command to execute CMake in the current directory (usually the build directory). This command will generate the Ninja build system based on the CMakeLists.txt file located in the parent directory. Once successfully executed, the project will be ready to be built using Ninja.
    ```bash
    cmake -G "Ninja" ..
    ```

**3. Build the project**  
  - Run the following command to build the project using the build system configured by CMake in the current directory. This will compile the project and create the executable SNU_BMT_GUI_Submitter.exe in the build folder.
    ```bash
    cmake --build .
    ```

**4. Setting Library Path for Executable in Current Directory**  
   - Run the following command to make the executable(SNU_BMT_GUI_Submitter) can reference the libraries located in the lib folder of the current directory.
     ```bash
     export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
     ```

**5. Start Performance Analysis**  
   - Run the following command to start created excutable. When the GUI Popup, Click [Start BMT] button to start AI Performance Analysis.
     ```bash
     ./SNU_BMT_GUI_Submitter
     ```
   

**build and start program example**
```bash
sjh@DESKTOP-U7I9FQS:~$ git clone https://github.com/kinsingo/SNU_BMT_GUI_Submitter_Linux.git
sjh@DESKTOP-U7I9FQS:~$ cd SNU_BMT_GUI_Submitter_Linux/build/
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ sudo apt install cmake
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ sudo apt install build-essential
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ sudo apt-get install ninja-build
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ sudo apt-get install libgl1 libgl1-mesa-dev
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ sudo sudo apt install unzip
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ rm -rf CMakeCache.txt CMakeFiles
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ cmake -G "Ninja" ..
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ cmake --build .
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ ./SNU_BMT_GUI_Submitter
```

**Run all commands at once (For Initial Build)**
```bash
git clone https://github.com/kinsingo/SNU_BMT_GUI_Submitter_Linux.git
cd SNU_BMT_GUI_Submitter_Linux/build/
sudo apt update
sudo apt install cmake
sudo apt install build-essential
sudo apt-get install ninja-build
sudo apt-get install libgl1 libgl1-mesa-dev
sudo apt install unzip
rm -rf CMakeCache.txt CMakeFiles
cmake -G "Ninja" ..
cmake --build .
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./SNU_BMT_GUI_Submitter
```

**Run all commands at once (For Rebuild)**
```bash
rm -rf CMakeCache.txt CMakeFiles
cmake -G "Ninja" ..
cmake --build .
export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
./SNU_BMT_GUI_Submitter
```


