//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#include <stdio.h>

#include "dhdc.h"



// macros
#define MIN(x,y) (x)<(y)?(x):(y)

// globals
int *ID;
int  Run = 0;



// gravity compensation thread
void *GravityThread (void *arg)
{
  // retrieve the device index as argument
  int id = *((int*)arg);

  // try and open requested device
  ID[id] = dhdOpenID (id);
  if (ID[id] < 0) {
    printf ("error: cannot open device (%s)\n", dhdErrorGetLastStr());
    Run = -1000;
    return NULL;
  }

  // report that we are ready
  Run++;

  // enable force
  dhdEnableForce (DHD_ON, ID[id]);

  // haptic loop
  while (Run > 0) {

    // apply zero force
    if (dhdSetForceAndTorqueAndGripperForce (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ID[id]) < DHD_NO_ERROR) {
      printf ("error: cannot set force (%s)\n", dhdErrorGetLastStr());
      Run = 0;
    }
  }

  // close the connection
  dhdClose (ID[id]);

  // report that we are done
  Run--;

  return NULL;
}



int
main (int  argc,
      char **argv)
{
  int    id = 0;
  int    count;
  double px, py, pz;

  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  printf ("\n");
  printf ("Force Dimension - Multi-threaded Gravity Compensation Example %d.%d.%d.%d\n", major, minor, release, revision);
  printf ("Copyright (C) 2001-2020 Force Dimension\n");
  printf ("All Rights Reserved.\n\n");

  // check for devices (limit to 10 devices)
  count = MIN(10, dhdGetDeviceCount ());
  if (count < 1) {
    printf ("error: no device detected\n");
    dhdSleep (2.0);
    return -1;
  }

  // allocate resources and start threads
  printf ("starting %d threads\n\n", count);
  ID = new int[count];
  for (int i=0; i<count; i++ ) {
    ID[i] = i;
    dhdStartThread (GravityThread, &(ID[i]), DHD_THREAD_PRIORITY_HIGH);
  }

  // wait for all threads to start (each thread increments the 'Run' variable by 1 when ready)
  while (Run >= 0 && Run < count) dhdSleep (0.1);
  if (Run < 0) {
    printf ("error: thread launch failed\n");
    return -1;
  }

  // identify each device
  for (int i=0;i <count; i++) printf ("[%d] %s device detected\n", i, dhdGetSystemName(ID[i]));
  printf ("\n");

  // display instructions
  printf ("press 'q' to quit\n");
  printf ("      [0..9] to select display device\n\n");

  // UI thread (default priority)
  while (Run > 0) {

    // display some info on the currently selected device
    dhdGetPosition (&px, &py, &pz, ID[id]);
    printf ("[%d] %s:  p (%+0.03f %+0.03f %+0.03f) m  |  freq [%0.02f kHz]\r", id, dhdGetSystemName(ID[id]), px, py, pz, dhdGetComFreq (ID[id]));

    // process user input (exit request or device selection)
    if (dhdKbHit()) {
      printf ("                                                                           \r");
      switch (dhdKbGet()) {
      case 'q': Run = 0; break;
      case '0': id  = MIN(count-1, 0); break;
      case '1': id  = MIN(count-1, 1); break;
      case '2': id  = MIN(count-1, 2); break;
      case '3': id  = MIN(count-1, 3); break;
      case '4': id  = MIN(count-1, 4); break;
      case '5': id  = MIN(count-1, 5); break;
      case '6': id  = MIN(count-1, 6); break;
      case '7': id  = MIN(count-1, 7); break;
      case '8': id  = MIN(count-1, 8); break;
      case '9': id  = MIN(count-1, 9); break;
      }
    }

    // we do not need to waste 100% of a CPU core on the UI thread
    dhdSleep (0.2);
  }

  // wait for all threads to finish
  while (Run > -count) dhdSleep (0.1);

  // happily exit
  printf ("\ndone.\n");
  return 0;
}
