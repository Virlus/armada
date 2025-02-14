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

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("Force Dimension - Automatic Initialization %d.%d.%d.%d\n", major, minor, release, revision);
  printf ("Copyright (C) 2001-2020 Force Dimension\n");
  printf ("All Rights Reserved.\n\n");

  // required to change asynchronous operation mode
  dhdEnableExpertMode ();

  // open the first available device
  if (drdOpen () < 0) {
    printf ("error: cannot open device (%s)\n", dhdErrorGetLastStr ());
    dhdSleep (2.0);
    return -1;
  }

  // print out device identifier
  if (!drdIsSupported()) {
    printf ("unsupported device\n");
    printf ("exiting...\n");
    dhdSleep (2.0);
    drdClose ();
    return -1;
  }
  printf ("%s haptic device detected\n\n", dhdGetSystemName());

  // perform auto-initialization
  printf ("initializing...\r");
  fflush (stdout);
  if (drdAutoInit () < 0) {
    printf ("error: auto-initialization failed (%s)\n", dhdErrorGetLastStr ());
    drdClose ();
    dhdSleep (2.0);
    return -1;
  }

  // perform initialization check (optional)
  printf ("checking initialization...\r");
  fflush (stdout);
  if (drdCheckInit () < 0) {
    printf ("error: device initialization check failed (%s)\n", dhdErrorGetLastStr ());
    drdClose ();
    dhdSleep (2.0);
    return -1;
  }

  // report success
  printf ("device successfully initialized\n\n");

  // stop regulation (and leave force enabled)
  drdStop (true);

  // display instructions
  printf ("press 'q' to quit\n\n");

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

      // retrieve information to display
      freq = dhdGetComFreq ();

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
      if (dhdKbHit() && dhdKbGet() == 'q') done = 1;
    }
  }

  // close the connection
  printf ("cleaning up...                                                           \n");
  drdClose ();

  // happily exit
  printf ("\ndone.\n");


  return 0;
}
