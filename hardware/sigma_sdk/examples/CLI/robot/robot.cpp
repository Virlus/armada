//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>

// some platforms do not define M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "drdc.h"



int
main (int  argc,
      char **argv)
{
  int    i;
  int    done;
  double px = 0.0, py = 0.0, pz = 0.0;
  double p[10];

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("Force Dimension - Robot Control Example %d.%d.%d.%d\n", major, minor, release, revision);
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
  printf ("press 'q' to quit demo\n\n");

  if (!drdIsInitialized() && dhdGetSystemType() == DHD_DEVICE_FALCON) {
    printf ("please initialize Falcon device...\r"); fflush(stdout);
    while (!drdIsInitialized()) dhdSetForce (0.0, 0.0, 0.0);
    printf ("                                  \r");
    dhdSleep (0.5);
  }

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

  // get workspace limits
  for (i=0; i<10; i++) p[i] = -0.04 + i*(0.08/10.0);

  // DEMO PART 1: move within the cubic robotic workspace at default accel/speed/decel
  //              (which are conservative) and with default precision (10 um)

  done = 0;
  while (done > -1 && done < 6) {

    // give the control thread some slack
    drdSleep (0.2);

    // generate random point within workspace
    while (!drdIsMoving ()) {
      px = 0.02 + p[rand()%10];
      py =        p[rand()%10];
      pz =        p[rand()%10];
      if (drdMoveToPos (px, py, pz, false) < 0) break;
      done++;
    }

    // check for exit condition
    if (dhdKbHit() && dhdKbGet()=='q') done = -1;

    // check that the control loop is still running
    if (!drdIsRunning ()) done = -2;

    // let user know what is going on
    printf ("control running at %0.03f kHz, moving to %0.02f, %0.02f, %0.02f      \r", drdGetCtrlFreq(), px, py, pz);
  }


  // DEMO PART 2: move to the extremities of the cubic robotic workspace,
  //              this time significantly faster

  // change motion generator parameters
  if (drdSetPosMoveParam (1.0, 5.0, 50.0) < 0) {
    printf ("warning: could not change regulation parameters\n");
  }

  if (done > -1) done = 0;
  while (done > -1 && done < 10) {

    // give the control thread some slack
    drdSleep (0.2);

    // generate random point within workspace
    while (!drdIsMoving ()) {
      px = 0.02 + p[rand()%10];
      py =        p[rand()%10];
      pz =        p[rand()%10];
      if (drdMoveToPos (px, py, pz, false) < 0) break;
      done++;
    }

    // check for exit condition
    if (dhdKbHit() && dhdKbGet()=='q') done = -1;

    // check that the control loop is still running
    if (!drdIsRunning ()) done = -2;

    // let user know what is going on
    printf ("control running at %0.03f kHz, moving to %0.02f, %0.02f, %0.02f      \r", drdGetCtrlFreq(), px, py, pz);
  }

  // reset motion control parameters
  if (drdSetPosMoveParam (1.0, 1.0, -1.0) < 0) {
    printf ("warning: could not change regulation parameters\n");
  }

  // DEMO PART 3: track a sphere in real-time
  //              this shows how to asynchronously send the robot along a control trajectory
  //              the trajectory is interpolated along the target points according to a set of physical parameters
  //              such as max allowed acceleration, max allowed velocity and max allowed deceleration time
  //              these parameters are controlled by the drdSetPoseTrackParam() function

  double dt;
  double t0;
  double x0 = 0.02;
  double y0 = 0.0;
  double z0 = 0.0;
  double radius = 0.03;
  double alpha, beta;
  double radial_vel   = 2*M_PI;
  double vertical_vel = M_PI/10.0;

  // move to the top of the sphere
  if (done > -1) drdMoveToPos (x0, y0, z0+radius);

  // enable trajectory interpolator and adjust its parameters
  drdEnableFilter (true);
  drdSetPosTrackParam (2.0, 1.0, -1.0);

  // start tracking a trajectory on the surface of the sphere
  // a new point along the trajectory is sent to the robotic controller at about 20 Hz
  // thanks to the trajectory interpolator, the movement is smooth

  t0 = drdGetTime();
  if (done > -1) done = 0;
  while (!done) {

    // generate next sphere target point
    dt    = drdGetTime()-t0;
    alpha = radial_vel * dt;
    beta  = M_PI/2.0 + (vertical_vel * dt);
    px    = x0+radius*cos(alpha)*cos(beta);
    py    = y0+radius*sin(alpha)*cos(beta);
    pz    = z0+radius*sin(beta);
    if (drdTrackPos (px, py, pz) < 0) break;
    if (beta > 5.0*M_PI/2.0) done = -1;

    // check for exit condition
    if (!drdIsRunning()) done = -2;
    if (dhdKbHit()) {
      if (dhdKbGet() == 'q') done = -1;
    }

    // throttle trajectory updates to 20 Hz
    printf ("control running at %0.03f kHz, tracing sphere surface...     \r", drdGetCtrlFreq());
    drdSleep (0.05);
  }

  // report control loop errors
  if (done == -2) printf ("error: robot control thread aborted (%s)\n", dhdErrorGetLastStr ());

  // close the connection
  printf ("cleaning up...                                             \n");
  drdClose ();

  // happily exit
  printf ("\ndone.\n");


  return 0;
}
