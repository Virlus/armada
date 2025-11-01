# ⚙️ Hardware setup

We select [Flexiv Rizon 4 robot](https://www.flexiv.com/products/rizon) as our robot hardware, equipped with [Robotiq 2F-85 gripper](https://robotiq.com/products/adaptive-grippers#Two-Finger-Gripper). 
The fingers are adapted from [UMI](https://umi-gripper.github.io/). 

We employ [Force Dimension sigma.7 haptic device](https://www.forcedimension.com/products/sigma) as our teleoperation interface. We also apply a [logitech G29 wheel](https://www.logitechg.com/en-us/shop/p/driving-force-racing-wheel) to enable larger workspace.

Two [Intel Realsense D435 cameras](https://www.intel.com/content/www/us/en/products/sku/128255/intel-realsense-depth-camera-d435/specifications.html) are deployed in our experiments, eye-in-hand and eye-to-hand respectively.

## 🦾 Robot hardware installation

We test our codebase on Ubuntu 20.04 and [Flexiv RDK v0.10](https://github.com/flexivrobotics/flexiv_rdk/tree/v0.10). The Flexiv Software Package has a version of `v2.11.5`.

It is strongly recommended that you install the C++ RDK follwing the [official guide](https://github.com/flexivrobotics/flexiv_rdk/tree/v0.10?tab=readme-ov-file#c-rdk). To validate that the installation is successful, you can check whether a shared object (e.g.`flexivrdk.cpython-310-x86_64-linux-gnu.so`) appears under the directory `flexiv_rdk/lib_py`. If so, place `flexiv_rdk` under [hardware/](./). 

The basic robot controller is implemented in [hardware/my_device/robot.py](./my_device/robot.py).

However, if your Flexiv robot has a newer version of software package (e.g. `v3.*`), we suggest checking the [latest Flexiv RDK](https://github.com/flexivrobotics/flexiv_rdk), which can be installed directly via Pip:

```
pip install flexivrdk
```

We also provide the CAD model for the adapter between [Robotiq 2F-85 gripper](https://robotiq.com/products/adaptive-grippers) and [UMI](https://github.com/real-stanford/universal_manipulation_interface) soft gripper in [hardware/my_device/ROBOTIQ.STEP](./my_device/ROBOTIQ.STEP).

## 🎮 Teleoperation device setup

### 🕹️ sigma.7 haptic device setup

The sigma.7 SDK files can be found [here](https://www.forcedimension.com/software/sdk), and the SDK directory should be placed under `hardware/sigma_sdk`. 

In order to obtain the shared object for sigma.7 control, first install `pybind11` and `cmake` via the following command:

```
pip install pybind11 cmake
```
After that, execute the following command under `hardware/sigma_sdk`:

```
c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) -I ./include sigma7.cpp -L ./lib/release/lin-x86_64-gcc/ -L /usr/local/lib/ -ldhd -ldrd -lpthread -lusb-1.0 -lrt -ldl -o sigma7$(python3-config --extension-suffix)
```

A shared object should appear under `hardware/sigma_sdk` (e.g. `sigma7.cpython-310-x86_64-linux-gnu.so`). After that, you can use the [basic controller for sigma.7.](./my_device/sigma.py).

### 🛞 logitech G29 wheel setup

The sigma.7 device has a confined workspace, thus restricting the robot from larger task workspace as well. To unlock full workspace for the robot, we deploy logitech G29 wheel to freeze the robot while sigma.7 device moves freely.
The freezing process can be activated by stepping on the right pedal of logitech G29 wheel, and the basic controller for it can be found [hardware/my_device/logitechG29_wheel.py](./my_device/logitechG29_wheel.py).

## 📷 Camera configuration

Please replace the camera serial numbers in [hardware macros](./my_device/macros.py) with that of your own Realsense cameras. In our experiments, we suggest putting the serial number of the eye-to-hand camera before that of the eye-in-hand camera.