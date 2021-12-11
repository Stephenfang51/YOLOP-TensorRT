# YOLOP-TensorRT

YOLOP is end2end NN used to detect object and segment drivable and lane segmentations, This is unofficial implementation of YOLOP-TensorRT, whereas the official code was no longer supported by author, and there are still bugs haven't been solved, I reconstruct the code and it is easy to use

most of module like bottleneckCSP etc are from **tensorrtx** which is great and useful.

### update

There are some tiny bugs still exist need to be fixed, still in process.

### requirements

- TensorRT (better >= 7)
- cuda (better >= 11)
- a powerful GPU or any Nvidia embedding device like Jetson Nano, NX

### installation

1. firstly modify CMakeLists.txt to adapt your enviroment, like cuda path, openCV
2. `mkdir build` & `cd build`
3. `cmake ..` & `make`

### how to use

1. convert yolop to wts using gen_wts.py, of course you need to download the pytorch model from official yolop
2. then `./build/buildengine`, you can add `-w` to add your wts file, `-b` to set batchsize, `-o` to set output engine
3. using `./build/inf` to inference imgs from directory or videos


![output](https://github.com/Stephenfang51/YOLOP-TensorRT/blob/main/output_demo_2.jpg)
