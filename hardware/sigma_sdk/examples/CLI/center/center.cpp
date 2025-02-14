//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "Eigen/Eigen"
using namespace Eigen;

#include "dhdc.h"
#include "drdc.h"

#define REFRESH_INTERVAL  0.1   // sec

#define KP    100.0
#define KVP    10.0
#define MAXF    4.0
#define KR      0.3
#define KWR     0.02
#define MAXT    0.1
#define KG    100.0
#define KVG     5.0
#define MAXG    1.0



int
main (int  argc,
      char **argv)
{
  double p[DHD_MAX_DOF];
  double r[3][3];
  double v[DHD_MAX_DOF];
  double f[DHD_MAX_DOF];
  double normf, normt;
  int    done  = 0;
  double t0    = dhdGetTime ();
  double t1    = t0;
  bool   base  = false;
  bool   wrist = false;
  bool   grip  = false;
  int    count = 0;
  int    res;

  // Eigen objects (mapped to the arrays above)
  Map<Vector3d> position(&p[0], 3);
  Map<Vector3d> force   (&f[0], 3);
  Map<Vector3d> torque  (&f[3], 3);
  Map<Vector3d> velpos  (&v[0], 3);
  Map<Vector3d> velrot  (&v[3], 3);
  Matrix3d center;
  Matrix3d rotation;

  // center of workspace
  center.setIdentity ();                           // rotation (matrix)
  double nullPose[DHD_MAX_DOF] = { 0.0, 0.0, 0.0,  // translation
                                   0.0, 0.0, 0.0,  // rotation (joint angles)
                                   0.0 };          // gripper

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("Force Dimension - Constant Centering Example %d.%d.%d.%d\n", major, minor, release, revision);
  printf ("Copyright (C) 2001-2020 Force Dimension\n");
  printf ("All Rights Reserved.\n\n");

  // open the first available device
  if (drdOpen () < 0) {
    printf ("error: cannot open device (%s)\n", dhdErrorGetLastStr ());
    dhdSleep (2.0);
    return -1;
  }

  // print out device identifier
  if (!drdIsSupported ()) {
    printf ("unsupported device\n");
    printf ("exiting...\n");
    dhdSleep (2.0);
    drdClose ();
    return -1;
  }
  printf ("%s haptic device detected\n\n", dhdGetSystemName ());

  // perform auto-initialization
  if (!drdIsInitialized () && drdAutoInit () < 0) {
    printf ("error: auto-initialization failed (%s)\n", dhdErrorGetLastStr ());
    dhdSleep (2.0);
    return -1;
  }
  else if (drdStart () < 0) {
    printf ("error: regulation thread failed to start (%s)\n", dhdErrorGetLastStr ());
    dhdSleep (2.0);
    return -1;
  }

  // move to center
  drdMoveTo (nullPose);

  // request a null force (only gravity compensation will be applied)
  // this will only apply to unregulated axis
  drdSetForceAndTorqueAndGripperForce (0.0, 0.0, 0.0,  // force
                                       0.0, 0.0, 0.0,  // torque
                                       0.0);           // gripper force

  // disable all axis regulation (but leave regulation thread running)
  drdRegulatePos  (base);
  drdRegulateRot  (wrist);
  drdRegulateGrip (grip);

  // display instructions
  printf ("press 'q' to quit\n");
  printf ("      'b' to free/hold base (translations)\n");
  printf ("      'w' to free/hold wrist (rotations)\n");
  printf ("      'g' to free/hold gripper\n\n");

  // display output header
  printf ("BASE  |  ");
  if (dhdHasActiveWrist ()) printf ("WRIST |  ");
  if (dhdHasGripper ())     printf ("GRIP  |  ");
  printf ("regulation  |  local   \n");
  printf ("---------");
  if (dhdHasActiveWrist ()) printf ("---------");
  if (dhdHasGripper ())     printf ("---------");
  printf ("-----------------------\n");

  // haptic loop
  while (!done) {

    // synchronize with regulation thread
    drdWaitForTick ();

    // get position/orientation/gripper and update Eigen rotation matrix
    drdGetPositionAndOrientation (&(p[0]), &(p[1]), &(p[2]),
                                  &(p[3]), &(p[4]), &(p[5]),
                                  &(p[6]), r);
    for (int i=0; i<3; i++) rotation.row(i) = Vector3d::Map(&r[i][0], 3);

    // get position/orientation/gripper velocity
    drdGetVelocity (&(v[0]), &(v[1]), &(v[2]),
                    &(v[3]), &(v[4]), &(v[5]),
                    &(v[6]));

    // compute base centering force
    force = - KP * position;

    // compute wrist centering torque
    AngleAxisd deltaRotation (rotation.transpose() * center);
    torque = rotation * (KR * deltaRotation.angle() * deltaRotation.axis());

    // compute gripper centering force
    f[6]  = - KG * (p[6] - 0.015);

    // scale force to a pre-defined ceiling
    if ((normf = force.norm()) > MAXF) force *= MAXF/normf;

    // scale torque to a pre-defined ceiling
    if ((normt = torque.norm()) > MAXT) torque *= MAXT/normt;

    // scale gripper force to a pre-defined ceiling
    if (f[6] >  MAXG) f[6] =  MAXG;
    if (f[6] < -MAXG) f[6] = -MAXG;

    // add damping
    force  -= KVP * velpos;
    torque -= KWR * velrot;
    f[6]   -= KVG * v[6];

    // apply centering force with damping
    res = drdSetForceAndTorqueAndGripperForce (f[0], f[1], f[2],  // force
                                               f[3], f[4], f[5],  // torque
                                               f[6]);             // gripper force
    if (res < DHD_NO_ERROR) {
      printf ("error: cannot set force (%s)\n", dhdErrorGetLastStr ());
      done = 1;
    }

    // local loop refresh rate
    count++;

    // display refresh rate and position at 10Hz
    t1 = drdGetTime ();
    if ((t1-t0) > REFRESH_INTERVAL) {

      // retrieve/compute information to display
      double freq = (double)count/(t1-t0)*1e-3;
      count = 0;
      t0 = t1;

      // print status
      if (base)    printf ("HOLD  |  ");
      else         printf ("free  |  ");
      if (dhdHasActiveWrist ()) {
        if (wrist) printf ("HOLD  |  ");
        else       printf ("free  |  ");
      }
      if (dhdHasGripper()) {
        if( grip)  printf ("HOLD  |  ");
        else       printf ("free  |  ");
      }
      printf ("  %0.2f kHz  |  %0.02f kHz       \r", dhdGetComFreq (), freq);

      // user input
      if (dhdKbHit ()) {
        switch (dhdKbGet ()) {
        case 'q': done = 1; break;
        case 'b':
          base = !base;
          drdRegulatePos  (base);
          break;
        case 'w':
          wrist = !wrist;
          drdRegulateRot  (wrist);
          break;
        case 'g':
          grip = !grip;
          drdRegulateGrip (grip);
          break;
        }
      }
    }
  }

  // stop regulation
  drdStop ();

  // close the connection
  printf ("cleaning up...                                                           \n");
  drdClose ();

  // happily exit
  printf ("\ndone.\n");


  return 0;
}
