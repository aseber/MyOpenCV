Note that all of these files require you to upgrade to OpenCV 3.0.0-dev.

You can do this by following this guide:
"http://www.bogotobogo.com/OpenCV/opencv_3_tutorial_ubuntu14_install_cmake.php"

The building process should take around 30 minutes, so be prepared.

Also, with 3.0.0 the OpenCV devs have moved around some files. You will no longer be able to use FAST or other proprietary detection algorithms unless you install this repository:
"https://github.com/Itseez/opencv_contrib"

Note that if you want to build the library with the contributed files, you will have to run the commands listed under the opencv_contrib readme documentation