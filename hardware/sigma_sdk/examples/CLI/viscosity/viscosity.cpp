//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#include <stdio.h>

#include "dhdc.h"

#define REFRESH_INTERVAL    0.1   //  seconds
#define LINEAR_VISCOSITY   20.0   //  N/(m/s)
#define ANGULAR_VISCOSITY   0.03  // Nm/(rad/s)



int
main (int  argc,
      char **argv)
{
  double vx, vy, vz;
  double wx, wy, wz;
  double vg;
  double fx, fy, fz;
  double tx, ty, tz;
  double fg;
  double t1,t0  = dhdGetTime ();
  int    done   = 0;

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("\n");
  printf ("Force Dimension - Viscosity Example %d.%d.%d.%d\n", major, minor, release, revision);
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

    dhdGetLinearVelocity (&vx, &vy, &vz);
    fx = -LINEAR_VISCOSITY * vx;
    fy = -LINEAR_VISCOSITY * vy;
    fz = -LINEAR_VISCOSITY * vz;

    dhdGetAngularVelocityRad (&wx, &wy, &wz);
    tx = -ANGULAR_VISCOSITY * wx;
    ty = -ANGULAR_VISCOSITY * wy;
    tz = -ANGULAR_VISCOSITY * wz;

    dhdGetGripperLinearVelocity (&vg);
    fg = -LINEAR_VISCOSITY * vg;

    // apply zero force
    if (dhdSetForceAndTorqueAndGripperForce (fx, fy, fz, tx, ty, tz, fg) < DHD_NO_ERROR) {
      printf ("error: cannot set force (%s)\n", dhdErrorGetLastStr());
      done = 1;
    }

    // display refresh rate and position at 10Hz
    t1 = dhdGetTime ();
    if ((t1-t0) > REFRESH_INTERVAL) {

      // update timestamp
      t0 = t1;

      // write down velocity
      printf ("v (%+0.03f %+0.03f %+0.03f) m/s  ", vx, vy, vz);
      if (dhdHasWrist   ()) printf ("|  w (%+02.01f %+02.01f %+02.01f) rad/s  ", wx, wy, wz);
      if (dhdHasGripper ()) printf ("|  vg (%+0.03f) m/s  ", vg);
      printf ("\r");

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
