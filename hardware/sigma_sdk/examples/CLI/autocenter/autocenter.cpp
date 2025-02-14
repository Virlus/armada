//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "dhdc.h"
#include "drdc.h"

#define REFRESH_INTERVAL  0.1   // sec



int
main (int  argc,
      char **argv)
{
  double px, py, pz;
  double fx, fy, fz;
  double freq   = 0.0;
  double t1,t0  = dhdGetTime ();
  int    done   = 0;

  // center of workspace
  double nullPose[DHD_MAX_DOF] = { 0.0, 0.0, 0.0,  // base  (translations)
                                   0.0, 0.0, 0.0,  // wrist (rotations)
                                   0.0 };          // gripper

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("Force Dimension - Auto Center Gravity %d.%d.%d.%d\n", major, minor, release, revision);
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

  // stop regulation thread (but leaves forces on)
  drdStop (true);

  // display instructions
  printf ("press 'q' to quit\n");
  printf ("      'c' to re-center end-effector (all axis)\n");
  printf ("      'p' to re-center position only\n");
  printf ("      'r' to re-center rotation only\n");
  printf ("      'g' to close gripper only\n\n");

  // haptic loop
  while (!done) {

    // apply zero force
    if (dhdSetForceAndTorqueAndGripperForce (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) < DHD_NO_ERROR) {
      printf ("error: cannot set force (%s)\n", dhdErrorGetLastStr ());
      done = 1;
    }

    // display refresh rate and position at 10Hz
    t1 = dhdGetTime ();
    if ((t1-t0) > REFRESH_INTERVAL) {

      // retrieve information to display
      freq = dhdGetComFreq ();
      t0   = t1;

      // write down position
      if (dhdGetPosition (&px, &py, &pz) < 0) {
        printf ("error: cannot read position (%s)\n", dhdErrorGetLastStr());
        done = 1;
      }
      if (dhdGetForce (&fx, &fy, &fz) < 0) {
        printf ("error: cannot read force (%s)\n", dhdErrorGetLastStr());
        done = 1;
      }
      printf ("p (%+0.03f %+0.03f %+0.03f) m  |  f (%+0.01f %+0.01f %+0.01f) N  |  freq [%0.02f kHz]       \r", px, py, pz, fx, fy, fz, freq);

      // user input
      if (dhdKbHit ()) {
        switch (dhdKbGet ()) {
        case 'q': done = 1; break;
        case 'c':
          drdRegulatePos  (true);
          drdRegulateRot  (true);
          drdRegulateGrip (true);
          drdStart();
          drdMoveTo (nullPose);
          drdStop(true);
          break;
        case 'p':
          drdRegulatePos  (true);
          drdRegulateRot  (false);
          drdRegulateGrip (false);
          drdStart();
          drdMoveToPos (0.0, 0.0, 0.0);
          drdStop(true);
          break;
        case 'r':
          drdRegulatePos  (false);
          drdRegulateRot  (true);
          drdRegulateGrip (false);
          drdStart();
          drdMoveToRot (0.0, 0.0, 0.0);
          drdStop(true);
          break;
        case 'g':
          drdRegulatePos  (false);
          drdRegulateRot  (false);
          drdRegulateGrip (true);
          drdStart();
          drdMoveToGrip (0.0);
          drdStop(true);
          break;
        }
      }
    }
  }

  // close the connection
  printf ("cleaning up...                                                           \n");
  drdClose ();

  // happily exit
  printf ("\ndone.\n");


  return 0;
}
