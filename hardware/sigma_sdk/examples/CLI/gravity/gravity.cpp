//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#include <stdio.h>

#include "dhdc.h"

#define REFRESH_INTERVAL  0.1   // sec



int
main (int  argc,
      char **argv)
{
  double px, py, pz;
  double fx, fy, fz;
  double t1,t0  = dhdGetTime ();
  int    done   = 0;

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("\n");
  printf ("Force Dimension - Gravity Compensation Example %d.%d.%d.%d\n", major, minor, release, revision);
  printf ("Copyright (C) 2001-2020 Force Dimension\n");
  printf ("All Rights Reserved.\n\n");

  // open the first available device
  if (dhdOpen () < 0) {
    printf ("error: cannot open device (%s)\n", dhdErrorGetLastStr());
    dhdSleep (2.0);
    return -1;
  }

  // identify device
  printf ("%s device detected\n\n", dhdGetSystemName());

  // display instructions
  printf ("press 'q' to quit\n\n");

  // enable force
  dhdEnableForce (DHD_ON);

  // haptic loop
  while (!done) {

    // apply zero force
    if (dhdSetForceAndTorqueAndGripperForce (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) < DHD_NO_ERROR) {
      printf ("error: cannot set force (%s)\n", dhdErrorGetLastStr());
      done = 1;
    }

    // display refresh rate and position at 10Hz
    t1 = dhdGetTime ();
    if ((t1-t0) > REFRESH_INTERVAL) {

      // update timestamp
      t0 = t1;

      // retrieve position
      if (dhdGetPosition (&px, &py, &pz) < DHD_NO_ERROR) {
        printf ("error: cannot read position (%s)\n", dhdErrorGetLastStr());
        done = 1;
      }

      // retrieve force
      if (dhdGetForce (&fx, &fy, &fz) < DHD_NO_ERROR) {
        printf ("error: cannot read force (%s)\n", dhdErrorGetLastStr());
        done = 1;
      }

      // display status
      printf ("p (%+0.03f %+0.03f %+0.03f) m  |  f (%+0.01f %+0.01f %+0.01f) N  |  freq %0.02f kHz\r", px, py, pz, fx, fy, fz, dhdGetComFreq());

      // user input
      if (dhdKbHit() && dhdKbGet() == 'q') done = 1;
    }
  }

  // close the connection
  dhdClose ();

  // happily exit
  printf ("\ndone.\n");
  return 0;
}
