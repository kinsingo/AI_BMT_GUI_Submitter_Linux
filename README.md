
## Submitter User Guide
**1. Implementation**
   1) Implement the overridden functions in the `Virtual_Submitter_Implementation` class within `main.cpp`  
   : Ensure that these functions are implemented to operate correctly in the intended calculator (e.g., CPU, GPU, NPU).

**2. Build**  
   1) Open the Linux terminal  
  -. Open the terminal (shell)  
  -. Navigate to the build directory using the command: [cd build]
   2) Generate the Ninja build system
  -. If cmake is not installed, Run the command [sudo apt install cmake]  
  -. If ninja is not installed, Run the command [sudo apt-get install ninja-build]  
  -. Run the command [cmake -G "Ninja" ..] to execute CMake in the current directory (usually the build directory). This command will generate the Ninja build system based on the CMakeLists.txt file located in the parent directory. Once successfully executed, the project will be ready to be built using Ninja.
   4) Build the project  
   : Run [cmake --build .] to build the project using the build system configured by CMake in the current directory. This will compile the project and create the executable SNU_BMT_GUI_Submitter.exe in the build folder.

## Build Example (Linux Terminal)
```bash
sjh@DESKTOP-U7I9FQS:~$ cd SNU_BMT_GUI_Submitter_Linux/
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux$ cd build
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ sudo apt-get install ninja-build
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ cmake -G "Ninja" ..
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ cmake --build .
sjh@DESKTOP-U7I9FQS:~/SNU_BMT_GUI_Submitter_Linux/build$ ./SNU_BMT_GUI_Submitter
