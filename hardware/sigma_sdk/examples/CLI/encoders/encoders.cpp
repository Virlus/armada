//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#include <stdio.h>

#include "dhdc.h"



int
main (int    argc,
      char **argv)
{
  int i;
  int done = 0;
  int enc[DHD_MAX_DOF];
  int encCount;

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("\n");
  printf ("Force Dimension - Encoder Reading Example %d.%d.%d.%d\n", major, minor, release, revision);
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

  // identify number of encoders to report based on device type
  switch (dhdGetSystemType ()) {
  case DHD_DEVICE_DELTA3:
  case DHD_DEVICE_OMEGA3:
  case DHD_DEVICE_FALCON:
    encCount = 3;
    break;
  case DHD_DEVICE_DELTA6:
  case DHD_DEVICE_OMEGA33:
  case DHD_DEVICE_OMEGA33_LEFT:
    encCount = 6;
    break;
  case DHD_DEVICE_OMEGA331:
  case DHD_DEVICE_OMEGA331_LEFT:
    encCount = 7;
    break;
  case DHD_DEVICE_CONTROLLER:
  case DHD_DEVICE_CONTROLLER_HR:
  default:
    encCount = 8;
    break;
  }

  // display instructions
  printf ("press 'q' to quit\n\n");
  printf ("encoder values:\n");

  // configure device
  dhdEnableExpertMode();

  // haptic loop
  while (!done) {

    // apply zero force for convenience
    dhdSetForceAndTorqueAndGripperForce (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

    // read all available encoders
    if (dhdGetEnc (enc) < 0) {
      printf ("error: cannot read encoders (%s)\n", dhdErrorGetLastStr ());
      done = 1;
    }

    // print out encoders according to system type
    for (i=0; i<encCount; i++) printf ("%06d ", enc[i]);
    printf ("          \r");

    // check for exit condition
    if (dhdKbHit() && dhdKbGet() == 'q') done = 1;
  }

  // close the connection
  dhdClose ();

  // happily exit
  printf ("\ndone.\n");
  return 0;
}
