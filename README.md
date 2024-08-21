## Submitter User Guide Steps
Step1) Build System Set-up  
Step2) Interface Implementation  
Step3) Build and Start BMT  

## Step 1) Build System Set-up (Installation Guide for Ubuntu)
**1. Update Your Package List**  
- Open a terminal and run the following command:
   ```bash
   sudo apt update
   ```

**2. Install CMake**  
- To install CMake, use the following command:
  ```bash
  sudo apt install cmake
  ```

**3. Install g++ compiler**  
- Install `build-essential` by running the following command:
  ```bash
  sudo apt install build-essential
  ```

**4. Install Ninja Build**  
- To install Ninja, run the following command:
  ```bash
  sudo apt-get install ninja-build
  ```

**5. Verify the Installation**  
- You can check the versions of the installed tools by running the following commands. If these commands return version information for each tool, the installation was successful.
  ```bash
  cmake --version
  gcc --version
  ninja --version
  ```

## Step2) Interface Implementation
- Implement the overridden functions in the `Virtual_Submitter_Implementation` class within `main.cpp`  
: Ensure that these functions are implemented to operate correctly in the intended calculator (e.g., CPU, GPU, NPU).

## Step3) Build and Start BMT
**1. Open the Window terminal**  
  - Navigate to the build directory using the following command
    ```bash
    cd SNU_BMT_GUI_Submitter_Windows\build
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

**4. Start builed BMT GUI**  
   - Run the following command to build the project using the build system configured by CMake in the current directory. This will compile the project and create the executable SNU_BMT_GUI_Submitter.exe in the build folder.
     ```bash
     .\SNU_BMT_GUI_Submitter
     ```

**build and start program example**
```bash
sjh@DESKTOP-U7I9FQS:~$ cd SNU_BMT_GUI_Submitter_Linux/build/
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ sudo apt install cmake
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ sudo apt install build-essential
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ sudo apt-get install ninja-build
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ rm -rf CMakeCache.txt CMakeFiles
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ cmake -G "Ninja" ..
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ cmake --build .
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ export LD_LIBRARY_PATH=$(pwd)/lib:$LD_LIBRARY_PATH
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ ./SNU_BMT_GUI_Submitter
```

