//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#include <stdio.h>

#include "dhdc.h"

#define REFRESH_INTERVAL  0.1   // sec



inline void
MatTranspose (const double a[3][3],
              double       m[3][3])
{
  m[0][0] = a[0][0];  m[0][1] = a[1][0];  m[0][2] = a[2][0];
  m[1][0] = a[0][1];  m[1][1] = a[1][1];  m[1][2] = a[2][1];
  m[2][0] = a[0][2];  m[2][1] = a[1][2];  m[2][2] = a[2][2];
}




int
main (int  argc,
      char **argv)
{
  const double K = 60;

  double px, py, pz;
  double fx, fy, fz;
  double j0, j1, j2;
  double g0, g1, g2;
  double q0, q1, q2;
  double J[3][3];
  double Jt[3][3];
  double freq   = 0.0;
  double t1,t0  = dhdGetTime ();
  bool   spring = false;
  int    done   = 0;
  int    sat;

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("Force Dimension - Jacobian Usage Example %d.%d.%d.%d\n", major, minor, release, revision);
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

  // emulate button on supported devices
  dhdEmulateButton (DHD_ON);

  // display instructions
  printf ("press BUTTON to enable virtual spring\n");
  printf ("         'q' to quit\n\n");

  // enable force
  dhdEnableForce (DHD_ON);

  // haptic loop
  while (!done) {

    // retrieve joint angles
    if (dhdGetPosition (&px, &py, &pz) < DHD_NO_ERROR) {
      printf ("error: cannot get joint angles (%s)\n", dhdErrorGetLastStr());
      done = 1;
    }

    // compute force to apply
    if (spring) {
      fx = - K * px;
      fy = - K * py;
      fz = - K * pz;
    }
    else fx = fy = fz = 0.0;

    // retrieve joint angles
    if (dhdGetDeltaJointAngles (&j0, &j1, &j2) < DHD_NO_ERROR) {
      printf ("error: cannot get joint angles (%s)\n", dhdErrorGetLastStr());
      done = 1;
    }

    // compute jacobian
    if (dhdDeltaJointAnglesToJacobian (j0, j1, j2, J) < DHD_NO_ERROR) {
      printf ("error: cannot compute jacobian (%s)\n", dhdErrorGetLastStr());
      done = 1;
    }

    // compute joint torques required for gravity compensation
    if (dhdDeltaGravityJointTorques (j0, j1, j2, &g0, &g1, &g2) < DHD_NO_ERROR) {
      printf ("error: cannot compute gravity compensation joint torques (%s)\n", dhdErrorGetLastStr());
      done = 1;
    }

    // compute joint torques Q = ((J)T) * F
    MatTranspose (J, Jt);
    q0 = Jt[0][0]*fx + Jt[0][1]*fy + Jt[0][2]*fz;
    q1 = Jt[1][0]*fx + Jt[1][1]*fy + Jt[1][2]*fz;
    q2 = Jt[2][0]*fx + Jt[2][1]*fy + Jt[2][2]*fz;

    // combine gravity compensation and requested force
    q0 += g0;
    q1 += g1;
    q2 += g2;

    // apply joint torques
    if ((sat = dhdSetDeltaJointTorques (q0, q1, q2)) < DHD_NO_ERROR) {
      printf ("error: cannot set joint torques (%s)\n", dhdErrorGetLastStr());
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
      if (sat == DHD_MOTOR_SATURATED) printf ("[*] ");
      else                            printf ("[-] ");
      printf ("q = (%+0.03f, %+0.03f, %+0.03f) [Nm]  |  freq = %0.02f [kHz]       \r", q0, q1, q2, freq);

      // user input
      if (dhdGetButtonMask()) spring = true;
      else                    spring = false;
      if (dhdKbHit() && dhdKbGet() == 'q') done = 1;
    }
  }

  // close the connection
  dhdClose ();

  // happily exit
  printf ("\ndone.\n");
  return 0;
}

