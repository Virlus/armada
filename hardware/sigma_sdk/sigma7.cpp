#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include <dhdc.h>
#include <drdc.h>
namespace py = pybind11;


PYBIND11_MODULE(sigma7, m) {
    m.doc() = "sigma7 python wrapper"; // optional module docstring

    // Robot SDK
    m.def(
        "drdOpen", 
        &drdOpen, 
        "Open a connection to the first compatible device connected to the system. To open connections to multiple devices, use the drdOpenID() call."
    );
    m.def(
        "drdOpenID", 
        &drdOpenID, 
        "Open a connection to one particular compatible device connected to the system. The order in which devices are opened persists until devices are added or removed. If the device at the specified index is already opened, its device ID is returned.", 
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetDevice", 
        &drdSetDevice, 
        "Select the default device that will receive the API commands. The API supports multiple devices. This routine allows the programmer to decide which device the API dhd_single_device_call single-device calls will address. Any subsequent API call that does not specifically mention the device ID in its parameter list will be sent to that device.", 
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetDeviceID", 
        &drdGetDeviceID, 
        "Return the ID of the current default device."
    );
    m.def(
        "drdClose", 
        &drdClose, 
        "Close the connection to a particular device.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdIsSupported", 
        &drdIsSupported, 
        "Determine if the device is supported out-of-the-box by the DRD. The regulation gains of supported devices are configured internally so that such devices are ready to use. Unsupported devices can still be operated with the DRD, but their regulation gains must first be configured using the drdSetEncPGain(), drdSetEncIGain() and drdSetEncDGain() functions.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdIsRunning", 
        &drdIsRunning, 
        "Checks the state of the robotic control thread for a particular device.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdIsMoving", 
        &drdIsMoving, 
        "Checks whether the particular robot is moving (following a call to drdMoveToPos(), drdMoveToEnc(), drdTrackPos() or drdTrackEnc()), as opposed to holding the target position after successfully reaching it.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdIsFiltering", 
        &drdIsFiltering, 
        "Checks whether the particular robot control thread is applying a motion filter while tracking a target using drdTrackPos() or drdTrackEnc().",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetTime", 
        &drdGetTime, 
        "Returns the current value from the high-resolution system counter in [s]. The resolution of the system counter may be machine-dependent, as it usually derived from one of the CPU clocks signals. The time returned is guaranteed to be monotonous."
    );
    m.def(
        "drdSleep", 
        &drdSleep, 
        "Suspend the calling thread for a given duration specified in [s]-> The sleep resolution is machine and OS dependent.",
        py::arg("sec")
    );
    m.def(
        "drdWaitForTick", 
        &drdWaitForTick, 
        "Synchronization function: calling this function will block until the next iteration of the control loop begins.",
        py::arg("ID") = static_cast<char>(0)        
    );
    m.def(
        "drdIsInitialized", 
        &drdIsInitialized, 
        "Checks the initialization status of a particular robot. The initialization status reflects the status of the controller RESET LED. The robot can be (re)initialized by calling drdAutoInit().",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdAutoInit", 
        &drdAutoInit, 
        "Performs automatic initialization of that particular robot by robotically moving to a known position and reseting encoder counters to their correct values.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdCheckInit", 
        &drdCheckInit, 
        "Check the validity of that particular robot initialization by robotically sweeping all endstops and comparing their joint space position to expected values (stored in each device internal memory). If the robot is not yet initialized, this function will first perform the same initialization routine as drdAutoInit() before running the endstop check.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetPositionAndOrientation", 
        [](char ID = static_cast<char>(0)) {
            double px = 0, py = 0, pz = 0, oa = 0, ob = 0, og = 0, pg = 0, _matrix[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            int sig = drdGetPositionAndOrientation(&px, &py, &pz, &oa, &ob, &og, &pg, _matrix, ID);
            auto matrix = py::array_t<double>({3, 3});
            auto r = matrix.mutable_unchecked<2>();
            for (size_t i = 0; i < 3; ++ i)
                for (size_t j = 0; j < 3; ++ j)
                    r(i, j) = _matrix[i][j];
            return std::make_tuple(sig, px, py, pz, oa, ob, og, pg, matrix);
        }, 
        "Retrieve the position of the end-effector in Cartesian coordinates. Please refer to your device user manual for more information on your device coordinate system.", 
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetVelocity", 
        [](char ID) {
            double vx = 0, vy = 0, vz = 0, wx = 0, wy = 0, wz = 0, vg = 0;
            int sig = drdGetVelocity(&vx, &vy, &vz, &wx, &wy, &wz, &vg, ID); 
            return std::make_tuple(sig, vx, vy, vz, wx, wy, wz, vg);
        }, 
        "Retrieve the velocity of the end-effector position in Cartesian coordinates. Please refer to your device user manual for more information on your device coordinate system.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetCtrlFreq", 
        &drdGetCtrlFreq, 
        "This function returns the average refresh rate of the control loop (in kHz) since the function was last called.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdStart", 
        &drdStart, 
        "Start the robotic control loop for the given robot. The robot must be initialized (either manually or with drdAutoInit()) before drdStart() can be called successfully.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdRegulatePos", 
        &drdRegulatePos, 
        "Enable/disable robotic regulation of the device delta base, which provides translations. If regulation is disabled, the base can move freely and will display any force set using drdSetForceAndTorqueAndGripperForce(). If it is enabled, base position is locked and can be controlled by calling all robotic functions (e.g. drdMoveToPos()). By default, delta base regulation is enabled.",
        py::arg("on"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdRegulateRot", 
        &drdRegulateRot, 
        "Enable/disable robotic regulation of the device wrist. If regulation is disabled, the wrist can move freely and will display any torque set using drdSetForceAndTorqueAndGripperForce(). If it is enabled, wrist orientation is locked and can be controlled by calling all robotic functions (e.g. drdMoveTo()). By default, wrist regulation is enabled.",
        py::arg("on"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdRegulateGrip", 
        drdRegulateGrip, 
        "Enable/disable robotic regulation of the device gripper. If regulation is disabled, the gripper can move freely and will display any force set using drdSetForceAndTorqueAndGripperForce(). If it is enabled, gripper orientation is locked and can be controlled by calling all robotic functions (e.g. drdMoveTo()). By default, gripper regulation is enabled.",
        py::arg("on"),
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetForceAndTorqueAndGripperForce", 
        &drdSetForceAndTorqueAndGripperForce, 
        "Apply force, torques and gripper force to all non-regulated, actuated DOFs of the device. The regulated DOFs can be selected using drdRegulatePos(), drdRegulateRot() and drdRegulateGrip(). The requested force is ignored for all regulated DOFs. You must use this function instead of all dhdSetForce() calls if the robotic regulation thread is running to prevent interfering with the regulation commands.",
        py::arg("fx"), py::arg("fy"), py::arg("fz"),
        py::arg("tx"), py::arg("ty"), py::arg("tz"),
        py::arg("fg"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetForceAndWristJointTorquesAndGripperForce", 
        &drdSetForceAndWristJointTorquesAndGripperForce, 
        "Apply force, wrist joint torques and gripper force to all non-regulated, actuated DOFs of the device. The regulated DOFs can be selected using drdRegulatePos(), drdRegulateRot() and drdRegulateGrip(). The requested force is ignored for all regulated DOFs. You must use this function instead of all dhdSetForce() calls if the robotic regulation thread is running to prevent interfering with the regulation commands.",
        py::arg("fx"), py::arg("fy"), py::arg("fz"),
        py::arg("t0"), py::arg("t1"), py::arg("t2"),
        py::arg("fg"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdMoveToPos", 
        &drdMoveToPos, 
        "Send the robot end-effector to a desired Cartesian position. The motion follows a straight line, with smooth acceleration/deceleration. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("px"), py::arg("py"), py::arg("pz"),
        py::arg("block"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdMoveToRot", 
        &drdMoveToRot, 
        "Send the robot end-effector to a desired Cartesian rotation. The motion follows a straight curve, with smooth acceleration/deceleration. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("oa"), py::arg("ob"), py::arg("og"),
        py::arg("block"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdMoveToGrip", 
        &drdMoveToGrip, 
        "Send the robot gripper to a desired opening distance. The motion is executed with smooth acceleration/deceleration. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("pg"), py::arg("block"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdMoveTo", 
        [](py::array_t<double> p, bool block, char ID = static_cast<char>(0)) {
            auto r = p.unchecked<1>();
            if (r.shape(0) != DHD_MAX_DOF)
                throw std::runtime_error("The length of array p must be the same as DHD_MAX_DOF.");
            double _p[DHD_MAX_DOF];
            for (int i = 0; i < DHD_MAX_DOF; ++ i)
                _p[i] = r(i);
            return drdMoveTo(_p, block, ID);
        }, 
        "Send the robot end-effector to a desired Cartesian 7-DOF configuration. The motion uses smooth acceleration/deceleration. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("p"), py::arg("block"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdMoveToEnc", 
        &drdMoveToEnc, 
        "Send the robot end-effector to a desired encoder position. The motion follows a straight line in the encoder space, with smooth acceleration/deceleration. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("enc0"), py::arg("enc1"), py::arg("enc2"),
        py::arg("block"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdMoveToAllEnc", 
        [](py::array_t<int> enc, bool block, char ID = static_cast<char>(0)) {
            auto r = enc.unchecked<1>();
            if (r.shape(0) != DHD_MAX_DOF)
                throw std::runtime_error("The length of array p must be the same as DHD_MAX_DOF.");
            int _enc[DHD_MAX_DOF];
            for (int i = 0; i < DHD_MAX_DOF; ++ i)
                _enc[i] = r(i);
            return drdMoveToAllEnc(_enc, block, ID);
        }, 
        "Send the robot end-effector to a desired encoder position. The motion follows a straight line in the encoder space, with smooth acceleration/deceleration. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("enc"), py::arg("block"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdTrackPos", 
        &drdTrackPos, 
        "Send the robot end-effector to a desired Cartesian position. If motion filters are enabled, the motion follows a smooth acceleration/deceleration constraint on each Cartesian axis. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("px"), py::arg("py"), py::arg("pz"),
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdTrackRot", 
        &drdTrackRot, 
        "Send the robot end-effector to a desired Cartesian orientation. If motion filters are enabled, the motion follows a smooth acceleration/deceleration curve along each Cartesian axis. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("oa"), py::arg("ob"), py::arg("og"),
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdTrackGrip", 
        &drdTrackGrip, 
        "Send the robot gripper to a desired opening distance. If motion filters are enabled, the motion follows a smooth acceleration/deceleration. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("pg"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdTrack", 
        [](py::array_t<double> p, char ID = static_cast<char>(0)) {
            auto r = p.unchecked<1>();
            if (r.shape(0) != DHD_MAX_DOF)
                throw std::runtime_error("The length of array p must be the same as DHD_MAX_DOF.");
            double _p[DHD_MAX_DOF];
            for (int i = 0; i < DHD_MAX_DOF; ++ i)
                _p[i] = r(i);
            return drdTrack(_p, ID);
        }, 
        "Send the robot end-effector to a desired Cartesian 7-DOF configuration. If motion filters are enabled, the motion follows a smooth acceleration/deceleration constraint on each Cartesian axis. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("p"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdTrackEnc", 
        &drdTrackEnc, 
        "Send the robot end-effector to a desired encoder position. If motion filters are enabled, the motion follows a smooth acceleration/deceleration constraint on each encoder axis. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("enc0"), py::arg("enc1"), py::arg("enc2"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdTrackAllEnc", 
        [](py::array_t<int> enc, char ID = static_cast<char>(0)) {
            auto r = enc.unchecked<1>();
            if (r.shape(0) != DHD_MAX_DOF)
                throw std::runtime_error("The length of array p must be the same as DHD_MAX_DOF.");
            int _enc[DHD_MAX_DOF];
            for (int i = 0; i < DHD_MAX_DOF; ++ i)
                _enc[i] = r(i);
            return drdTrackAllEnc(_enc, ID);
        },
        "Send the robot end-effector to a desired encoder position. If motion filters are enabled, the motion follows a smooth acceleration/deceleration constraint on each encoder axis. The acceleration and velocity profiles can be controlled by adjusting the trajectory generation parameters.",
        py::arg("enc"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdHold", 
        &drdHold, 
        "Immediately make the robot hold its current position. All motion commands are abandoned.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdLock", 
        &drdLock, 
        "Depending on the value of the mask parameter, either move the device to its park position and engage the locks, or remove the locks. This function only applies to devices equipped with mechanical locks, and will return an error when called on other devices.",
        py::arg("mask"), py::arg("init"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdStop", 
        &drdStop, 
        "Stop the robotic control loop for the given robot.",
        py::arg("frc"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetPriorities", 
        [](char ID = static_cast<char>(0)) {
            int prio = 0, ctrlprio = 0;
            int sig = drdGetPriorities(&prio, &ctrlprio, ID);
            return std::make_tuple(sig, prio, ctrlprio);
        }, 
        "This function makes it possible to retrieve the priority of the control thread and the calling thread. Thread priority is system dependent, as described in thread priorities.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetPriorities", 
        &drdSetPriorities, 
        "This function makes it possible to adjust the priority of the control thread and the calling thread. Thread priority is system dependent, as described in thread priorities.",
        py::arg("prio"), py::arg("ctrlprio"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetEncPGain", 
        &drdSetEncPGain, 
        "Set the P term of the PID controller that regulates the base joint positions. In practice, this affects the stiffness of the regulation.",
        py::arg("gain"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetEncPGain", 
        &drdGetEncPGain, 
        "Retrieve the P term of the PID controller that regulates the base joint positions.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetEncIGain",
        &drdSetEncIGain, 
        "Set the I term of the PID controller that regulates the base joint positions. In practice, this affects the precision of the regulation.",
        py::arg("gain"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetEncIGain", 
        &drdGetEncIGain, 
        "Retrieve the I term of the PID controller that regulates the base joint positions.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetEncDGain", 
        &drdSetEncDGain, 
        "Set the D term of the PID controller that regulates the base joint positions. In practice, this affects the velocity of the regulation.",
        py::arg("gain"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetEncDGain", 
        &drdGetEncDGain, 
        "Retrieve the D term of the PID controller that regulates the base joint positions.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetMotRatioMax", 
        &drdSetMotRatioMax, 
        "Set the maximum joint torque applied to all regulated joints expressed as a fraction of the maximum torque available for each joint. In practice, this limits the maximum regulation torque (in joint space), making it potentially safer to operate in environments where humans or delicate obstacles are present.",
        py::arg("scale"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetMotRatioMax", 
        &drdGetMotRatioMax, 
        "Retrieve the maximum joint torque applied to all regulated joints expressed as a fraction of the maximum torque available for each joint.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetEncMoveParam", 
        &drdSetEncMoveParam, 
        "Set encoder positioning trajectory generation parameters.",
        py::arg("amax"), py::arg("vmax"), py::arg("jerk"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetEncTrackParam", 
        &drdSetEncTrackParam, 
        "Set encoder tracking trajectory generation parameters.",
        py::arg("amax"), py::arg("vmax"), py::arg("jerk"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetPosMoveParam", 
        &drdSetPosMoveParam, 
        "Set Cartesian positioning trajectory generation parameters.",
        py::arg("amax"), py::arg("vmax"), py::arg("jerk"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdSetPosTrackParam", 
        &drdSetPosTrackParam, 
        "Set Cartesian tracking trajectory generation parameters.",
        py::arg("amax"), py::arg("vmax"), py::arg("jerk"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetEncMoveParam", 
        [](char ID = static_cast<char>(0)) {
            double amax = 0, vmax = 0, jerk = 0;
            int sig = drdGetEncMoveParam(&amax, &vmax, &jerk, ID);
            return std::make_tuple(sig, amax, vmax, jerk);
        },
        "Retrieve encoder positioning trajectory generation parameters.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetEncTrackParam", 
        [](char ID = static_cast<char>(0)) {
            double amax = 0, vmax = 0, jerk = 0;
            int sig = drdGetEncTrackParam(&amax, &vmax, &jerk, ID);
            return std::make_tuple(sig, amax, vmax, jerk);
        },
        "Retrieve encoder tracking trajectory generation parameters.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetPosMoveParam", 
        [](char ID = static_cast<char>(0)) {
            double amax = 0, vmax = 0, jerk = 0;
            int sig = drdGetPosMoveParam(&amax, &vmax, &jerk, ID);
            return std::make_tuple(sig, amax, vmax, jerk);
        }, 
        "Retrieve Cartesian positioning trajectory generation parameters.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "drdGetPosTrackParam", 
        [](char ID = static_cast<char>(0)) {
            double amax = 0, vmax = 0, jerk = 0;
            int sig = drdGetPosTrackParam(&amax, &vmax, &jerk, ID);
            return std::make_tuple(sig, amax, vmax, jerk);
        }, 
        "Retrieve Cartesian tracking trajectory generation parameters.",
        py::arg("ID") = static_cast<char>(0)
    );

    // Errors
    m.def(
        "dhdErrorGetLast", 
        &dhdErrorGetLast, 
        "Returns the last error code encountered in the running thread. See error management for details."
    );
    m.def(
        "dhdErrorGetLastStr", 
        &dhdErrorGetLastStr, 
        "Returns a brief string describing the last error encountered in the running thread. See error management for details."
    );
    m.def(
        "dhdErrorGetStr", 
        &dhdErrorGetStr, 
        "Returns a brief string describing a given error code. See error management for details.",
        py::arg("error")
    );

    // standard SDK
    m.def(
        "dhdEnableSimulator",
        &dhdEnableSimulator, 
        "Enable device simulator support. This enables network access on the loopback interface.",
        py::arg("on")
    );
    m.def(
        "dhdGetDeviceCount", 
        &dhdGetDeviceCount, 
        "Return the number of compatible Force Dimension devices connected to the system. This encompasses all devices connected locally, including devices already locked by other applications. Devices are given a unique identifier, as explained in the multiple devices section."
    );
    m.def(
        "dhdGetAvailableCount", 
        &dhdGetAvailableCount, 
        "Return the number of available Force Dimension devices connected to the system. This encompasses all devices connected locally, but excludes devices already locked by other applications. Devices are given a unique identifier, as explained in the multiple devices section."
    );
    m.def(
        "dhdSetDevice", 
        &dhdSetDevice, 
        "Select the default device that will receive the SDK commands. The SDK supports multiple devices. This routine allows the programmer to decide which device the SDK dhd_single_device_call single-device calls will address. Any subsequent SDK call that does not specifically mention the device ID in its parameter list will be sent to that device.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetDeviceID", 
        &dhdGetDeviceID, 
        "Return the ID of the current default device."
    );
    m.def(
        "dhdGetSerialNumber", 
        [](char ID = static_cast<char>(0)) {
            unsigned short sn;
            int sig = dhdGetSerialNumber(&sn, ID);
            return std::make_tuple(sig, sn);
        },
        "Return the device serial number.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdOpen", 
        &dhdOpen, 
        "Open a connection to the first available device connected to the system. The order in which devices are opened persists until devices are added or removed."
    );
    m.def(
        "dhdOpenType", 
        &dhdOpenType, 
        "Open a connection to the first device of a given type connected to the system. The order in which devices are opened persists until devices are added or removed.",
        py::arg("type")
    );
    m.def(
        "dhdOpenSerial", 
        &dhdOpenSerial, 
        "Open a connection to the device with a given serial number (available on recent models only).",
        py::arg("serial")
    );
    m.def(
        "dhdOpenID", 
        &dhdOpenID, 
        "Open a connection to one particular device connected to the system. The order in which devices are opened persists until devices are added or removed. If the device at the specified index is already opened, its device ID is returned.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdClose", 
        &dhdClose, 
        "Close the connection to a particular device.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdStop", 
        &dhdStop, 
        "Stop the device. This routine disables the force on the haptic device and puts it into BRAKE mode.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetComMode", 
        &dhdGetComMode, 
        "Retrieve the COM operation mode on compatible devices.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdEnableForce", 
        &dhdEnableForce, 
        "Enable the force mode in the device controller. val=1 is on, val=0 is off.",
        py::arg("val"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdEnableGripperForce", 
        &dhdEnableGripperForce, 
        "Enable the gripper force mode in the device controller. val=1 is on, val=0 is off. This function is only relevant to devices that have a gripper with a default closed or opened state. It does not apply to the sigma.x and omega.x range of devices, whose gripper does not have a default state. For those devices, the gripper force is enabled/disabled by dhdEnableForce().",
        py::arg("val"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetSystemType", 
        &dhdGetSystemType, 
        "Return the haptic device type. As this SDK can be used to control all of Force Dimension haptic products, this can help programmers ensure that their application is running on the appropriate target haptic device.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetSystemName", 
        &dhdGetSystemName, 
        "Return the haptic device type. As this SDK can be used to control all of Force Dimension haptic products, this can help programmers ensure that their application is running on the appropriate target haptic device.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetVersion", 
        [](char ID = static_cast<char>(0)) {
            double ver = 0;
            int sig = dhdGetVersion(&ver, ID);
            return std::make_tuple(sig, ver);
        }, 
        "Return the device controller version. As this SDK can be used to control all of Force Dimension haptic products, this can help programmers ensure that their application is running on the appropriate version of the haptic controller.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetSDKVersion", 
        [](char ID = static_cast<char>(0)) {
            int major = 0, minor = 0, release = 0, revision = 0;
            dhdGetSDKVersion(&major, &minor, &release, &revision);
            return std::make_tuple(major, minor, release, revision);
        }, 
        "Return the SDK complete set of version numbers.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetStatus", 
        [](char ID = static_cast<char>(0)) {
            int _status[DHD_MAX_STATUS];
            int sig = dhdGetStatus(_status, ID);
            auto status = py::array_t<double>({DHD_MAX_STATUS});
            auto r = status.mutable_unchecked<1>();
            for (size_t i = 0; i < DHD_MAX_STATUS; ++ i)
                r(i) = _status[i];
            return std::make_tuple(sig, status);
        }, 
        "Returns the status vector of the haptic device. The status is described in the status section.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetDeviceAngleRad", 
        [](char ID = static_cast<char>(0)) {
            double angle = 0;
            int sig = dhdGetDeviceAngleRad(&angle, ID);
            return std::make_tuple(sig, angle);
        }, 
        "Get the device base plate angle around the Y axis.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetDeviceAngleDeg", 
        [](char ID = static_cast<char>(0)) {
            double angle = 0;
            int sig = dhdGetDeviceAngleDeg(&angle, ID);
            return std::make_tuple(sig, angle);
        }, 
        "Get the device base plate angle around the Y axis.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetEffectorMass", 
        [](char ID = static_cast<char>(0)) {
            double mass = 0;
            int sig = dhdGetEffectorMass(&mass, ID);
            return std::make_tuple(sig, mass);
        }, 
        "Get the mass of the end-effector currently defined for a device. The gripper mass is used in the gravity compensation feature.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetSystemCounter", 
        &dhdGetSystemCounter, 
        "Returns a timestamp computed from the high-resolution system counter, expressed in microseconds. This function is deprecated, please use dhdGetTime() instead."
    );
    m.def(
        "dhdGetButton", 
        &dhdGetButton, 
        "Return the status of the button located on the end-effector. 0 is pressed, 1 is not, -1 is error",
        py::arg("index"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetButtonMask", 
        &dhdGetButtonMask, 
        "Return the 32-bit binary mask of the device buttons.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdSetOutput", 
        &dhdSetOutput, 
        "Set the user programmable output bits on devices that support it.",
        py::arg("output"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdIsLeftHanded", 
        &dhdIsLeftHanded, 
        "Returns true if the device is configured for left-handed use, false otherwise.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdHasBase", 
        &dhdHasBase, 
        "Returns true if the device has a base, false otherwise.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdHasWrist", 
        &dhdHasWrist, 
        "Returns true if the device has a wrist, false otherwise.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdHasActiveWrist", 
        &dhdHasActiveWrist, 
        "Returns true if the device has an active wrist, false otherwise.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdHasGripper", 
        &dhdHasGripper, 
        "Returns true if the device has a gripper, false otherwise.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdHasActiveGripper",
        &dhdHasActiveGripper, 
        "Returns true if the device has an active wrist, false otherwise.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdReset", 
        &dhdReset, 
        "Puts the device in RESET mode.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdResetWrist", 
        &dhdResetWrist, 
        "(deprecated)",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdWaitForReset",
        &dhdWaitForReset, 
        "Puts the device in RESET mode and wait for the user to calibrate the device. Optionally, a timeout can be defined after which the call returns even if calibration has not occurred.",
        py::arg("timeout"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdSetStandardGravity", 
        &dhdSetStandardGravity, 
        "Set the standard gravity constant used in gravity compensation. By default, the constant is set to 9.81 m/s^2.",
        py::arg("g"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdSetGravityCompensation", 
        &dhdSetGravityCompensation, 
        "Enable/disable gravity compensation.",
        py::arg("val"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdSetBrakes", 
        &dhdSetBrakes, 
        "Enable/disable the device electromagnetic brakes.",
        py::arg("val"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdSetDeviceAngleRad", 
        &dhdSetDeviceAngleRad, 
        "Set the device base plate angle around the (inverted) Y axis. Please refer to your device user manual for more information on your device coordinate system. An angle value of 0 corresponds to the device \"upright\" position, with its base plate perpendicular to axis X. An angle value of Pi/2 corresponds to the device base plate resting horizontally.",
        py::arg("angle"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdSetDeviceAngleDeg", 
        &dhdSetDeviceAngleDeg, 
        "Set the device base plate angle around the (inverted) Y axis. Please refer to your device user manual for more information on your device coordinate system. An angle value of 0 corresponds to the device \"upright\" position, with its base plate perpendicular to axis X. An angle value of 90 corresponds to the device base plate resting horizontally.",
        py::arg("angle"), py::arg("ID") = static_cast<char>(0)       
    );
    m.def(
        "dhdSetEffectorMass", 
        &dhdSetEffectorMass, 
        "Define the mass of the end-effector. This function is required to provide accurate gravity compensation when custom-made or modified end-effectors are used.",
        py::arg("mass"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetPosition", 
        [](char ID = static_cast<char>(0)) {
            double px = 0, py = 0, pz = 0;
            int sig = dhdGetPosition(&px, &py, &pz, ID);
            return std::make_tuple(sig, px, py, pz);
        }, 
        "Retrieve the position of the end-effector in Cartesian coordinates. Please refer to your device user manual for more information on your device coordinate system.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetForce", 
        [](char ID = static_cast<char>(0)) {
            double fx = 0, fy = 0, fz = 0;
            int sig = dhdGetForce(&fx, &fy, &fz, ID);
            return std::make_tuple(sig, fx, fy, fz);
        },
        "Retrieve the force vector applied to the end-effector.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdSetForce", 
        &dhdSetForce, 
        "Set the desired force vector in Cartesian coordinates to be applied to the end-effector of the device.",
        py::arg("fx"), py::arg("fy"), py::arg("fz"), py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetOrientationRad", 
        [](char ID = static_cast<char>(0)) {
            double oa = 0, ob = 0, og = 0;
            int sig = dhdGetOrientationRad(&oa, &ob, &og, ID);
            return std::make_tuple(sig, oa, ob, og);
        }, 
        "For devices with a wrist structure, retrieve individual angle of each joint, starting with the one located nearest to the wrist base plate. For the DHD_DEVICE_OMEGA33 and DHD_DEVICE_OMEGA33_LEFT devices, angles are computed with respect to their internal reference frame, which is rotated 45 degrees around the Y axis. Please refer to your device user manual for more information on your device coordinate system.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetOrientationDeg", 
        [](char ID = static_cast<char>(0)) {
            double oa = 0, ob = 0, og = 0;
            int sig = dhdGetOrientationDeg(&oa, &ob, &og, ID);
            return std::make_tuple(sig, oa, ob, og);
        }, 
        "For devices with a wrist structure, retrieve individual angle of each joint, starting with the one located nearest to the wrist base plate. For the DHD_DEVICE_OMEGA33 and DHD_DEVICE_OMEGA33_LEFT devices, angles are computed with respect to their internal reference frame, which is rotated 45 degrees around the Y axis. Please refer to your device user manual for more information on your device coordinate system.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetPositionAndOrientationRad", 
        [](char ID = static_cast<char>(0)) {
            double px = 0, py = 0, pz = 0, oa = 0, ob = 0, og = 0;
            int sig = dhdGetPositionAndOrientationRad(&px, &py, &pz, &oa, &ob, &og, ID);
            return std::make_tuple(sig, px, py, pz, oa, ob, og);
        }, 
        "Retrieve the position and orientation of the end-effector in Cartesian coordinates. For devices with a wrist structure, the orientation is expressed as the individual angle of each joint, starting with the one located nearest to the wrist base plate. For the DHD_DEVICE_OMEGA33 and DHD_DEVICE_OMEGA33_LEFT devices, angles are computed with respect to their internal reference frame, which is rotated 45 degrees around the Y axis. Please refer to your device user manual for more information on your device coordinate system.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetPositionAndOrientationDeg", 
        [](char ID = static_cast<char>(0)) {
            double px = 0, py = 0, pz = 0, oa = 0, ob = 0, og = 0;
            int sig = dhdGetPositionAndOrientationDeg(&px, &py, &pz, &oa, &ob, &og, ID);
            return std::make_tuple(sig, px, py, pz, oa, ob, og);
        }, 
        "Retrieve the position and orientation of the end-effector in Cartesian coordinates. For devices with a wrist structure, the orientation is expressed as the individual angle of each joint, starting with the one located nearest to the wrist base plate. For the DHD_DEVICE_OMEGA33 and DHD_DEVICE_OMEGA33_LEFT devices, angles are computed with respect to their internal reference frame, which is rotated 45 degrees around the Y axis. Please refer to your device user manual for more information on your device coordinate system.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetPositionAndOrientationFrame", 
        [](char ID = static_cast<char>(0)) {
            double px = 0, py = 0, pz = 0, _matrix[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            int sig = dhdGetPositionAndOrientationFrame(&px, &py, &pz, _matrix, ID);
            auto matrix = py::array_t<double>({3, 3});
            auto r = matrix.mutable_unchecked<2>();
            for (size_t i = 0; i < 3; ++ i)
                for (size_t j = 0; j < 3; ++ j)
                    r(i, j) = _matrix[i][j];
            return std::make_tuple(sig, px, py, pz, matrix);
        }, 
        "Retrieve the position and orientation matrix of the end-effector in Cartesian coordinates. Please refer to your device user manual for more information on your device coordinate system.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetForceAndTorque", 
        [](char ID = static_cast<char>(0)) {
            double fx = 0, fy = 0, fz = 0, tx = 0, ty = 0, tz = 0;
            int sig = dhdGetForceAndTorque(&fx, &fy, &fz, &tx, &ty, &tz, ID);
            return std::make_tuple(sig, fx, fy, fz, tx, ty, tz);
        }, 
        "Retrieve the force and torque vectors applied to the device end-effector, as well as the force applied to the gripper.",
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdSetForceAndTorque", 
        &dhdSetForceAndTorque, 
        "Set the desired force and torque vectors to be applied to the device end-effector.",
        py::arg("fx"), py::arg("fy"), py::arg("fz"), 
        py::arg("tx"), py::arg("ty"), py::arg("tz"),
        py::arg("ID") = static_cast<char>(0)
    );
    m.def(
        "dhdGetOrientationFrame", 
        [](char ID = static_cast<char>(0)) {
            double _matrix[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            int sig = dhdGetOrientationFrame(_matrix, ID);
            auto matrix = py::array_t<double>({3, 3});
            auto r = matrix.mutable_unchecked<2>();
            for (size_t i = 0; i < 3; ++ i)
                for (size_t j = 0; j < 3; ++ j)
                    r(i, j) = _matrix[i][j];
            return std::make_tuple(sig, matrix);
        }, 
        "Retrieve the rotation matrix of the wrist structure. The identity matrix is returned for devices that do not support orientations.",
        py::arg("ID") = static_cast<char>(0)        
    );
    m.def(
        "dhdGetGripperAngleDeg", 
        [](char ID = static_cast<char>(0)) {
            double a = 0;
            int sig = dhdGetGripperAngleDeg(&a, ID);
            return std::make_tuple(sig, a);
        }, 
        "Get the gripper opening angle in degrees.",
        py::arg("ID") = static_cast<char>(0)     
    );
    m.def("dhdGetGripperAngleRad", &dhdGetGripperAngleRad, "");
    m.def("dhdGetGripperGap", &dhdGetGripperGap, "");
    m.def("dhdGetGripperThumbPos", &dhdGetGripperThumbPos, "");
    m.def("dhdGetGripperFingerPos", &dhdGetGripperFingerPos, "");
    m.def("dhdGetComFreq", &dhdGetComFreq, "");
    m.def("dhdSetForceAndGripperForce", &dhdSetForceAndGripperForce, "");
    m.def("dhdSetForceAndTorqueAndGripperForce", &dhdSetForceAndTorqueAndGripperForce, "");
    m.def("dhdGetForceAndTorqueAndGripperForce", &dhdGetForceAndTorqueAndGripperForce, "");
    m.def("dhdConfigLinearVelocity", &dhdConfigLinearVelocity, "");
    m.def("dhdGetLinearVelocity", &dhdGetLinearVelocity, "");
    m.def("dhdConfigAngularVelocity", &dhdConfigAngularVelocity, "");
    m.def("dhdGetAngularVelocityRad", &dhdGetAngularVelocityRad, "");
    m.def("dhdGetAngularVelocityDeg", &dhdGetAngularVelocityDeg, "");
    m.def("dhdConfigGripperVelocity", &dhdConfigGripperVelocity, "");
    m.def("dhdGetGripperLinearVelocity", &dhdGetGripperLinearVelocity, "");
    m.def("dhdGetGripperAngularVelocityRad", &dhdGetGripperAngularVelocityRad, "");
    m.def("dhdGetGripperAngularVelocityDeg", &dhdGetGripperAngularVelocityDeg, "");
    m.def("dhdEmulateButton", &dhdEmulateButton, "");
    m.def("dhdGetBaseAngleXRad", &dhdGetBaseAngleXRad, "");
    m.def("dhdGetBaseAngleXDeg", &dhdGetBaseAngleXDeg, "");
    m.def("dhdSetBaseAngleXRad", &dhdSetBaseAngleXRad, "");
    m.def("dhdSetBaseAngleXDeg", &dhdSetBaseAngleXDeg, "");
    m.def("dhdGetBaseAngleZRad", &dhdGetBaseAngleZRad, "");
    m.def("dhdGetBaseAngleZDeg", &dhdGetBaseAngleZDeg, "");
    m.def("dhdSetBaseAngleZRad", &dhdSetBaseAngleZRad, "");
    m.def("dhdSetBaseAngleZDeg", &dhdSetBaseAngleZDeg, "");
    m.def("dhdSetVibration", &dhdSetVibration, "");
    m.def("dhdSetMaxForce", &dhdSetMaxForce, "");
    m.def("dhdSetMaxTorque", &dhdSetMaxTorque, "");
    m.def("dhdSetMaxGripperForce", &dhdSetMaxGripperForce, "");
    m.def("dhdGetMaxForce", &dhdGetMaxForce, "");
    m.def("dhdGetMaxTorque", &dhdGetMaxTorque, "");
    m.def("dhdGetMaxGripperForce", &dhdGetMaxGripperForce, "");

    // expert SDK
    m.def("dhdEnableExpertMode", &dhdEnableExpertMode, "");
    m.def("dhdDisableExpertMode", &dhdDisableExpertMode, "");
    m.def("dhdDisableExpertMode", &dhdDisableExpertMode, "");
    m.def("dhdCalibrateWrist", &dhdCalibrateWrist, "");
    m.def("dhdSetTimeGuard", &dhdSetTimeGuard, "");
    m.def("dhdSetVelocityThreshold", &dhdSetVelocityThreshold, "");
    m.def("dhdGetVelocityThreshold", &dhdGetVelocityThreshold, "");
    m.def("dhdUpdateEncoders", &dhdUpdateEncoders, "");
    m.def("dhdGetDeltaEncoders", &dhdGetDeltaEncoders, "");
    m.def("dhdGetWristEncoders", &dhdGetWristEncoders, "");
    m.def("dhdGetGripperEncoder", &dhdGetGripperEncoder, "");
    m.def("dhdGetEncoder", &dhdGetEncoder, "");
    m.def("dhdSetMotor", &dhdSetMotor, "");
    m.def("dhdSetDeltaMotor", &dhdSetDeltaMotor, "");
    m.def("dhdSetWristMotor", &dhdSetWristMotor, "");
    m.def("dhdSetGripperMotor", &dhdSetGripperMotor, "");
}
