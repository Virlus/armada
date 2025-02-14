//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "drdc.h"



int
main (int  argc,
      char **argv)
{
  int    done = 0;
  double f, fx, fy, fz;
  double ofx = 0.0, ofy = 0.0, ofz = 0.0;
  double sx, sy, sz;
  double Pgain;
  double PgainStep;
  bool   oldButton = false;

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("Force Dimension - Force Sensor Emulation %d.%d.%d.%d\n", major, minor, release, revision);
  printf ("Copyright (C) 2001-2020 Force Dimension\n");
  printf ("All Rights Reserved.\n\n");

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

  // display instructions
  printf ("press BUTTON to move sensor point\n");
  printf ("press ' ' to calibrate sensor\n");
  printf ("press ',' to decrease Pgain (stiffness)\n");
  printf ("      '.' to increase Pgain (stiffness)\n");
  printf ("      'q' to quit demo\n\n");

  // initialize if necessary
  if (!drdIsInitialized() && (drdAutoInit() < 0)) {
    printf ("error: initialization failed (%s)\n", dhdErrorGetLastStr ());
    dhdSleep (2.0);
    return -1;
  }

  // start robot control loop
  if (drdStart() < 0) {
    printf ("error: control loop failed to start properly (%s)\n", dhdErrorGetLastStr ());
    dhdSleep (2.0);
    return -1;
  }

  // goto workspace center
  if (drdMoveToPos (0.0, 0.0, 0.0, false) < 0) {
    printf ("error: failed to move to central position (%s)\n", dhdErrorGetLastStr ());
    dhdSleep (2.0);
    return -1;
  }

  // retrieve current regulation gain (joint-space spring stiffness)
  Pgain = drdGetEncPGain ();
  PgainStep = 0.01*Pgain;

  // disable integral term (we do not need positional precision)
  drdSetEncIGain (0.0);

  // display output format
  printf (" Pgain  |   force [N]  |      fx       fy       fz  [N]\n");
  printf ("--------|--------------|-------------------------------\n");

  // continuously measure and display force while DRD holds position in the background
  while (!done) {

    // measure force
    dhdGetForce (&fx, &fy, &fz);
    sx = fx - ofx;
    sy = fy - ofy;
    sz = fz - ofz;
    f = sqrt(sx*sx+sy*sy+sz*sz);

    // display force
    printf ("  %4d  |      %+6.02f  |  %+6.02f   %+6.02f   %+6.02f   \r", (int)Pgain, f, sx, sy, sz);

    // check for exit condition
    if (!drdIsRunning()) done = -1;
    if (dhdKbHit()) {
      switch (dhdKbGet()) {
      case ' ': ofx = fx; ofy = fy; ofz = fz; break;
      case 'q': done = 1; break;
      case ',': Pgain = Pgain-PgainStep; break;
      case '.': Pgain = Pgain+PgainStep; break;
      }

      // adjust Pgain
      if (Pgain < 0.0) Pgain = 0.0;
      drdSetEncPGain (Pgain);
    }

    // we do not need to run fast
    dhdSleep (0.01);

    // move the regulation point if end-effector button is depressed
    bool button = (dhdGetButton (0) != 0);
    if ( button && !oldButton) drdRegulatePos (false);
    if (!button &&  oldButton) {
      drdHold ();
      drdRegulatePos (true);
    }
    oldButton = button;
  }

  // report control loop errors
  if (done == -1) printf ("error: robot control thread aborted (%s)\n", dhdErrorGetLastStr ());

  // close the connection
  printf ("cleaning up...                                             \n");
  drdClose ();

  // happily exit
  printf ("\ndone.\n");


  return 0;
}
