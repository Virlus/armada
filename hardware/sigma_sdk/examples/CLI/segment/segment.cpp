//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#include <stdio.h>
#include <math.h>

#include "dhdc.h"

#define REFRESH_INTERVAL  0.05   // sec



// Compute the projection of a point 'p' onto a segment
// defined by two points 'A' and 'B'.
//
// Projection is returned in point 's'.

void
project_point_on_segment (double p[3],
                          double A[3],
                          double B[3],
                          double s[3])
{
  double Ap[3];
  double AB[3];
  double dir[3];

  // compute relevant vectors
  for (int i=0; i<3; i++) {
    AB[i] = B[i] - A[i];
    Ap[i] = p[i] - A[i];
  }

  // compute segment norm, return if too small
  double norm = sqrt (AB[0]*AB[0] + AB[1]*AB[1] + AB[2]*AB[2]);
  if (norm <= 1e-6) {
    for (int i=0; i<3; i++) s[i] = p[i];
    return;
  }

  // compute segment direction unit vector
  for (int i=0; i<3; i++) dir[i] = AB[i] / norm;

  // compute projection ratio
  double proj = Ap[0]*dir[0] + Ap[1]*dir[1] + Ap[2]*dir[2];

  // compute point projection on segment
  if      (proj < 0.0)  for (int i=0; i<3; i++) s[i] = A[i];
  else if (proj > norm) for (int i=0; i<3; i++) s[i] = B[i];
  else                  for (int i=0; i<3; i++) s[i] = A[i]+proj*dir[i];
}



// compute the projection of a force 'g' onto a direction
// defined by two points 'p' and 's'.
//
// Projection is returned in force 'f'.

void
project_force_on_direction (double g[3],
                            double p[3],
                            double s[3],
                            double f[3])
{
  double ps[3];
  double dir[3];

  // compute unit direction vector, return if norm is too small
  for (int i=0; i<3; i++) ps[i] = s[i] - p[i];
  double norm = sqrt (ps[0]*ps[0] + ps[1]*ps[1] + ps[2]*ps[2]);
  if (norm < 1e-6) {
    for (int i=0; i<3; i++) f[i] = 0.0;
    return;
  }
  for (int i=0; i<3; i++) dir[i] = ps[i]/norm;

  // compute force projection
  double proj = g[0]*dir[0] + g[1]*dir[1] + g[2]*dir[2];
  for (int i=0; i<3; i++) f[i] = proj * dir[i];
}



// entry point

int
main (int  argc,
      char **argv)
{
  int    done  = 0;
  double t1,t0 = dhdGetTime ();

  double p[3];
  double v[3];
  double s[3];
  double g[3];
  double f[3];

  bool   oldButton  = false;
  int    pointCount = 0;
  double point[2][3];

  const double K = 2000.0;
  const double C = 20.0;

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("\n");
  printf ("Force Dimension - Segment Constraint Example %d.%d.%d.%d\n", major, minor, release, revision);
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
  printf ("press BUTTON or 'p' to define segment points\n");
  printf ("      'c' to clear current segment constraint\n");
  printf ("      'q' to quit\n\n");

  // enable force and button emulation, disable velocity threshold
  dhdEnableExpertMode ();
  dhdSetVelocityThreshold (0);
  dhdEnableForce (DHD_ON);
  dhdEmulateButton (DHD_ON);

  // loop while the user does not press 'q'
  while (!done) {

    // get position and velocity
    dhdGetPosition       (&(p[0]), &(p[1]), &(p[2]));
    dhdGetLinearVelocity (&(v[0]), &(v[1]), &(v[2]));

    // if a segment has been defined
    if (pointCount >= 2) {

      // compute projection of the device position onto segment
      project_point_on_segment (p, point[0], point[1], s);

      // compute guidance force, modeled as a spring+damper system that pulls
      // the device towards its projection on the constraint segment
      for (int i=0; i<3; i++) g[i] = K * (s[i] - p[i]) - C * v[i];

      // project guidance force onto the vector defined by the device position and its projection;
      // this removes all unwanted force components (e.g. viscosity along the "free" direction)
      project_force_on_direction (g, p, s, f);
    }

    // otherwise, apply a null force
    else f[0] = f[1] = f[2] = 0.0;

    // apply force
    if (dhdSetForceAndTorqueAndGripperForce (f[0], f[1], f[2], 0.0, 0.0, 0.0, 0.0) < DHD_NO_ERROR) {
      printf ("error: cannot set force (%s)\n", dhdErrorGetLastStr());
      done = 1;
    }

    // user interface
    t1 = dhdGetTime ();
    if ((t1-t0) > REFRESH_INTERVAL) {

      bool button   = false;
      bool setPoint = false;

      // check for user input (button and keyboard)
      if (dhdGetButton (0) == DHD_ON) button   = true;
      if (oldButton && !button)       setPoint = true;
      oldButton = button;
      if (dhdKbHit()) {
        switch (dhdKbGet()) {
        case 'q': done       = 1;     break;
        case 'p': setPoint   = true;  break;
        case 'c': pointCount = 0;     break;
        }
      }

      // define point if requested
      if (pointCount < 2 && setPoint) {
        point[pointCount][0] = p[0];
        point[pointCount][1] = p[1];
        point[pointCount][2] = p[2];
        pointCount++;
        if (pointCount == 2) printf ("Press 'c' clear                    \r");
      }

      // display instructions
      if (pointCount == 0) printf ("Please define first segment point  \r");
      if (pointCount == 1) printf ("Please define second segment point \r");

      // update timestamp
      t0 = t1;
    }
  }

  // close the connection
  dhdClose ();

  // happily exit
  printf ("done.                              \n");
  return 0;
}
