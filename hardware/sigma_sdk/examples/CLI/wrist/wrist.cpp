//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>

#include "dhdc.h"

#define REFRESH_INTERVAL  0.1   // sec
#define R2D(x) 180.0 * (x) / M_PI



int
main (int  argc,
      char **argv)
{
  // regulation stiffness
  const double Kj[DHD_MAX_DOF] = { 0.0, 0.0, 0.0,
                                   4.0, 3.0, 1.0,    // locked wrist joint stiffness (in Nm/rad)
                                   0.0,
                                   0.0 };

  // regulation viscosity
  const double Kv[DHD_MAX_DOF] = { 0.0, 0.0, 0.0,
                                   0.04, 0.03, 0.01, // locked wrist joint viscosity (in Nm/(rad/s))
                                   0.0,
                                   0.0 };

  double jointAngle[DHD_MAX_DOF];
  double jointVelocity[DHD_MAX_DOF];
  double jointAngleTarget[DHD_MAX_DOF];
  double jointTorque[DHD_MAX_DOF];
  bool   lock[DHD_MAX_DOF];
  double t1,t0 = dhdGetTime ();
  int    done  = 0;

  // initialize variables
  for (int i = 0; i < DHD_MAX_DOF; i++) lock[i] = 0.0;

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("\n");
  printf ("Force Dimension - Active Wrist Locking Example %d.%d.%d.%d\n", major, minor, release, revision);
  printf ("Copyright (C) 2001-2020 Force Dimension\n");
  printf ("All Rights Reserved.\n\n");

  // required to access joint angles
  dhdEnableExpertMode ();

  // open the first available device
  if (dhdOpen () < 0) {
    printf ("error: cannot open device (%s)\n", dhdErrorGetLastStr());
    dhdSleep (2.0);
    return -1;
  }

  // identify device
  printf ("%s device detected\n\n", dhdGetSystemName());

  // check that we have
  if (!dhdHasActiveWrist ()) {
    dhdClose ();
    printf ("error: device does not have an active wrist\n");
    dhdSleep (2.0);
    return -1;
  }

  // display instructions
  printf ("press 'q' to quit\n");
  printf ("      '0' to toggle virtual lock on wrist joint 0\n");
  printf ("      '1' to toggle virtual lock on wrist joint 1\n");
  printf ("      '2' to toggle virtual lock on wrist joint 2\n");
  printf ("      'a' to toggle virtual lock on all wrist joints\n\n");

  // enable force
  dhdEnableForce (DHD_ON);

  // haptic loop
  while (!done) {

    // retrieve wrist joints
    dhdGetJointAngles     (jointAngle);
    dhdGetJointVelocities (jointVelocity);

    // compute joint torques as appropriate
    for (int i=3; i<6; i++) {
      if (lock[i]) jointTorque[i] = -Kj[i] * (jointAngle[i] - jointAngleTarget[i]) - Kv[i] * jointVelocity[i];
      else         jointTorque[i] = 0.0;
    }

    // apply as appropriate
    int res = dhdSetForceAndWristJointTorquesAndGripperForce (0.0, 0.0, 0.0,
                                                              jointTorque[3], jointTorque[4], jointTorque[5],
                                                              0.0);
    if (res < DHD_NO_ERROR) {
      printf ("error: cannot set force (%s)\n", dhdErrorGetLastStr());
      done = 1;
    }

    // display refresh rate and position at 10Hz
    t1 = dhdGetTime ();
    if ((t1-t0) > REFRESH_INTERVAL) {

      // update timestamp
      t0 = t1;

      // display status
      printf ("j (%+0.03f %+0.03f %+0.03f) m  |  r (%+04.0f %+04.0f %+04.0f) Nmm  |  freq %0.02f kHz   \r",
              R2D(jointAngle[3]), R2D(jointAngle[4]), R2D(jointAngle[5]), jointTorque[3]*1e3, jointTorque[4]*1e3, jointTorque[5]*1e3,
              dhdGetComFreq());

      // user input
      if (dhdKbHit()) {
        switch (dhdKbGet()) {
        case 'q': done = 1; break;
        case 'a':
          lock[3] = lock[4] = lock[5] = !lock[3];
          if (lock[3]) for (int i=3; i<6; i++) jointAngleTarget[i] = jointAngle[i];
          break;
        case '0':
          lock[3] = !lock[3];
          jointAngleTarget[3] = jointAngle[3];
          break;
        case '1':
          lock[4] = !lock[4];
          jointAngleTarget[4] = jointAngle[4];
          break;
        case '2':
          lock[5] = !lock[5];
          jointAngleTarget[5] = jointAngle[5];
          break;
        }
      }
    }
  }

  // close the connection
  dhdClose ();

  // happily exit
  printf ("\ndone.\n");
  return 0;
}
