//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <string.h>

#include <iostream>
using namespace std;

#include "CMacrosGL.h"
using namespace Eigen;

#include "dhdc.h"

// GLFW library
#include <GLFW/glfw3.h>



// constants
const double CONST_SMALL = 1e-10;

// size of torus
const float TorusRadius0 = 0.05f;
const float TorusRadius1 = 0.027f;

// size of finger
const double FingerRadius = 0.005;

// status flags
bool SimulationOn;
bool SimulationFinished;

// device-specific globals
int        NumDevice;
char      *DeviceID;
int       *FingerCount;
Vector3d **FingerPosGlobal;
Matrix3d  *DeviceRotGlobal;
bool      *HasRot;
bool      *HasGrip;

// position and orientation of torus in global world coordinates
Vector3d TorusPosGlobal;
Matrix3d TorusRotGlobal;
Vector3d TorusVelGlobal;

// global force
Vector3d ForceGlobal[2];

// text overlay globals
double LastTime = dhdGetTime();
double Freq;
char   Perf[50];
bool   ShowRate = true;

// object properties
const double Stiffness = 1000.0;
const double Mass      =  100.0;
const double Viscosity =    1.0;

// GLFW display globals
GLFWwindow *window          = NULL;
int         width           = 0;
int         height          = 0;
int         swapInterval    = 1;



// render torus
void
DrawTorus (float majorRadius,
           float minorRadius,
           int   numMajor,
           int   numMinor)
{
  double majorStep = 2.0 * M_PI / numMajor;
  double minorStep = 2.0 * M_PI / numMinor;

  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

  for (int i = 0; i < numMajor; ++i) {
    double a0 = i * majorStep;
    double a1 = a0 + majorStep;
    GLdouble x0 = cos(a0);
    GLdouble y0 = sin(a0);
    GLdouble x1 = cos(a1);
    GLdouble y1 = sin(a1);

    glBegin(GL_TRIANGLE_STRIP);

    for (int j = 0; j <= numMinor; ++j) {
      double b = j * minorStep;
      GLdouble c = cos(b);
      GLdouble r = minorRadius * c + majorRadius;
      GLdouble z = minorRadius * sin(b);

      glNormal3d(x0 * c, y0 * c, z / minorRadius);
      glTexCoord2d(i / (GLdouble) numMajor, j / (GLdouble) numMinor);
      glVertex3d(x0 * r, y0 * r, z);

      glNormal3d(x1 * c, y1 * c, z / minorRadius);
      glTexCoord2d((i + 1) / (GLdouble) numMajor, j / (GLdouble) numMinor);
      glVertex3d(x1 * r, y1 * r, z);
    }

    glEnd();
  }
}



// OpenGL rendering
void
UpdateGraphics (void)
{
  if (SimulationOn) {

    int d, i;

    cMatrixGL      mat;
    GLUquadricObj *sphere;

    // clean things up
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // render torus
    GLfloat mat_ambient0[]  = {0.1f, 0.1f, 0.3f};
    GLfloat mat_diffuse0[]  = {0.1f, 0.3f, 0.5f};
    GLfloat mat_specular0[] = {0.5f, 0.5f, 0.5f};
    glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient0);
    glMaterialfv (GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse0);
    glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular0);
    glMaterialf  (GL_FRONT_AND_BACK, GL_SHININESS, 1.0);

    mat.set (TorusPosGlobal, TorusRotGlobal);
    mat.glMatrixPushMultiply ();
    DrawTorus (TorusRadius0, TorusRadius1, 64, 64);
    mat.glMatrixPop ();

    // render cursors
    GLfloat mat_ambient1[]  = {0.4f, 0.4f, 0.4f};
    GLfloat mat_diffuse1[]  = {0.8f, 0.8f, 0.8f};
    GLfloat mat_specular1[] = {0.5f, 0.5f, 0.5f};
    glMaterialfv (GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient1);
    glMaterialfv (GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse1);
    glMaterialfv (GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular1);
    glMaterialf  (GL_FRONT_AND_BACK, GL_SHININESS, 1.0);

    for (d=0; d<NumDevice; d++) {

      for (i=0; i<FingerCount[d]; i++) {
        mat.set (FingerPosGlobal[d][i], DeviceRotGlobal[d]);
        mat.glMatrixPushMultiply ();
        sphere = gluNewQuadric ();
        gluSphere (sphere, FingerRadius, 32, 32);

        // render a small frame
        if (HasRot[d]) {
          glDisable  (GL_LIGHTING);
          glBegin    (GL_LINES);
          glColor3f (0.45f, 0.45f, 0.45f);
          glVertex3d (0.00,  0.000,  0.000);
          glVertex3d (0.02,  0.000,  0.000);
          glVertex3d (0.02, -0.004,  0.000);
          glVertex3d (0.02,  0.004,  0.000);
          glVertex3d (0.02,  0.000, -0.004);
          glVertex3d (0.02,  0.000,  0.004);
          glEnd();
          glEnable  (GL_LIGHTING);
        }

        mat.glMatrixPop ();
      }
    }

    // text overlay
    if (ShowRate) {
      if (dhdGetTime() - LastTime > 0.1) {
        Freq     = dhdGetComFreq(DeviceID[0]);
        LastTime = dhdGetTime ();
        sprintf (Perf, "%0.03f kHz", Freq);
      }
      glDisable (GL_LIGHTING);
      glColor3f  (1.0, 1.0, 1.0);
      glRasterPos3f (0.0f, -0.01f, -0.1f);
      for (char *c=Perf; *c != '\0'; c++) renderBitmapCharacter(*c, HELVETICA12);
      glEnable  (GL_LIGHTING);
    }

    GLenum err = glGetError();
    if (err != GL_NO_ERROR) printf ("error:  %s\n", gluErrorString(err));

  }
}



// exit callback
void
Close (void)
{
  // finish haptic loop
  SimulationOn = false;
  while (!SimulationFinished) dhdSleep (0.1);

  // close devices
  for (int i=0; i<NumDevice; i++) dhdClose ();
}



// haptic thread
void*
HapticsLoop (void* pUserData)
{
  static double t0 = dhdGetTime ();
  double        t  = t0 + 0.001;
  double        dt;
  bool          initialized = false;

  int      i,d;
  double   r[3][3];
  Vector3d rawPosGlobal[2];
  Vector3d fingerPosLocal;
  Vector3d forceLocal;

  Vector3d radius[2];
  radius[0] << 0.0, -0.02, 0.0;
  radius[1] << 0.0,  0.02, 0.0;

  // start haptic simulation
  SimulationOn       = true;
  SimulationFinished = false;

  // start with no force
  ForceGlobal[0].setZero ();
  ForceGlobal[1].setZero ();

  // enable force
  for (d=0; d<NumDevice; d++) dhdEnableForce (DHD_ON, DeviceID[d]);

  // main haptic simulation loop
  while (SimulationOn) {

    t  = dhdGetTime ();
    dt = t - t0;
    t0 = t;

    for (d=0; d<NumDevice; d++) {

      double    x,y,z;

      // select active device
      dhdSetDevice (DeviceID[d]);

      // retrieve orientation frame (identity for 3-dof devices)
      dhdGetOrientationFrame (r);
      DeviceRotGlobal[d]  << r[0][0], r[0][1], r[0][2],
                             r[1][0], r[1][1], r[1][2],
                             r[2][0], r[2][1], r[2][2];

      // adapt behavior to device capabilities
      if (HasGrip[d]) {
        dhdGetGripperThumbPos (&x,&y,&z);
        rawPosGlobal[0] << x,y,z;

        dhdGetGripperFingerPos (&x,&y,&z);
        rawPosGlobal[1] << x,y,z;
      }
      else {
        dhdGetPosition (&x, &y, &z);
        rawPosGlobal[0] << x,y,z;
      }

      // compute interaction between torus and each finger
      for (i=0; i<FingerCount[d]; i++) {

        // init variables
        ForceGlobal[i].setZero ();

        // compute position of finger in local coordinates of torus
        fingerPosLocal = TorusRotGlobal.transpose() * (rawPosGlobal[i] - TorusPosGlobal);

        // project finger on torus plane (z=0)
        Vector3d fingerProjection = fingerPosLocal;
        fingerProjection(2) = 0;

        // search for the nearest point on the torus medial axis
        if (fingerPosLocal.squaredNorm() > CONST_SMALL) {
          Vector3d pointAxisTorus = TorusRadius0 * fingerProjection.normalized();

          // compute eventual penetration of finger inside the torus
          Vector3d vectTorusFinger = fingerPosLocal - pointAxisTorus;

          // finger inside torus, compute forces
          double distance = vectTorusFinger.norm();
          if ((distance < (TorusRadius1 + FingerRadius)) && (distance > 0.001)) {
            forceLocal = ((TorusRadius1 + FingerRadius) - distance) * Stiffness * vectTorusFinger.normalized();
            fingerPosLocal = pointAxisTorus + (TorusRadius1 + FingerRadius) * vectTorusFinger.normalized();
          }

          // finger is outside torus
          else forceLocal.setZero();
        }
        else forceLocal.setZero();

        // convert reaction force and position in world coordinates
        ForceGlobal[i]        = TorusRotGlobal * forceLocal;
        FingerPosGlobal[d][i] = TorusRotGlobal * fingerPosLocal;

        // update angular velocity
        TorusVelGlobal += -1.0/Mass * dt * (FingerPosGlobal[d][i] - TorusPosGlobal).cross (ForceGlobal[i]);
      }

      // compute projected force on each gripper finger
      Vector3d force;
      double    gripperForce = 0.0;
      if (HasGrip) {

        // compute total force
        force = ForceGlobal[0] + ForceGlobal[1];
        Vector3d gripdir = rawPosGlobal[1] - rawPosGlobal[0];

        // if force is not null
        if (gripdir.norm() > 0.00001) {

          // project force on mobile gripper finger (forceGlobal[1]) onto gripper opening vector (gripdir)
          gripdir.normalize ();
          Vector3d gripper = (ForceGlobal[1].dot (gripdir) / (gripdir.squaredNorm())) * gripdir;
          gripperForce = gripper.norm();

          // compute the direction of the force based on the angle between
          // the gripper force vector (gripper) and the gripper opening vector (gripdir)
          if (force.norm() > 0.001) {
            double cosangle = gripdir.dot (gripper) / (gripdir.norm()*gripper.norm());
            if      (cosangle >  1.0) cosangle =  1.0;
            else if (cosangle < -1.0) cosangle = -1.0;
            double angle = acos(cosangle);
            if ((angle > M_PI/2.0) || (angle < -M_PI/2.0)) gripperForce = -gripperForce;
          }
        }

        // invert if necessary for left-handed devices
        if (dhdIsLeftHanded()) gripperForce = -gripperForce;
      }
      else force = ForceGlobal[0];

      // make sure we don't fly off if our initial position is inside the torus
      if (!initialized) {
        if (force.norm() == 0.0 && gripperForce == 0.0) initialized = true;
        else {
          force.setZero();
          gripperForce = 0.0;
        }
      }

      // apply all forces at once
      dhdSetForceAndGripperForce (force(0), force(1), force(2), gripperForce);
    }

    // add damping or freeze if any of the devices button is pressed
    for (d=0; d<NumDevice; d++) if (dhdGetButtonMask(DeviceID[d]) & 0x02) TorusVelGlobal.setZero();
    TorusVelGlobal *= (1.0 - Viscosity*dt);

    // compute next pose of torus
    if (TorusVelGlobal.norm() > CONST_SMALL) {
      Matrix3d torusRotationIncrement;
      torusRotationIncrement = AngleAxisd (TorusVelGlobal.norm(), TorusVelGlobal.normalized());
      TorusRotGlobal = torusRotationIncrement * TorusRotGlobal;
    }
  }

  // close connection with haptic device
  for (d=0; d<NumDevice; d++) dhdClose (DeviceID[d]);

  // simulation is now exiting
  SimulationFinished = true;

  // return
  return NULL;
}



// window resize GLFW callback
void
OnWindowResize(GLFWwindow *window,
               int         w,
               int         h)
{
  double glAspect = ((double)w / (double)h);

  width  = w;
  height = h;

  glViewport(0, 0, w, h);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(60, glAspect, 0.01, 10);

  gluLookAt(0.2, 0, 0,
            0, 0, 0,
            0, 0, 1);

  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
}



// GLFW window keypress callback
void
OnKey (GLFWwindow* window,
       int         key,
       int         scancode,
       int         action,
       int         mods)
{
  // filter out calls that only include a key press
  if (action != GLFW_PRESS) return;

  // detect exit requests
  if ((key == GLFW_KEY_Q) || (key == GLFW_KEY_Q)) {
    SimulationOn = false;
    while (!SimulationFinished) dhdSleep (0.01);
    exit(0);
  }

  // toggle refresh rate display
  if (key == GLFW_KEY_R) ShowRate = !ShowRate;
}



// GLFW error callback
void
OnError (int a_error, const char* a_description)
{
    cout << "Error: " << a_description << endl;
}



// GLFW initialization
int
InitGLFW ()
{
  // initialize GLFW library
  if (!glfwInit()) return -1;

  // set error callback
  glfwSetErrorCallback (OnError);

  // compute desired size of window
  const GLFWvidmode* mode = glfwGetVideoMode (glfwGetPrimaryMonitor());
  int w = (int)(0.8 *  mode->height);
  int h = (int)(0.5 *  mode->height);
  int x = (int)(0.5 * (mode->width  - w));
  int y = (int)(0.5 * (mode->height - h));

  // configure OpenGL rendering
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_DOUBLEBUFFER, GLFW_TRUE);
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_STEREO, GL_FALSE);
  glfwWindowHint(GLFW_VISIBLE, GL_FALSE);

  // create window and display context
  window = glfwCreateWindow(w, h, "Force Dimension - OpenGL Torus Example", NULL, NULL);
  if (!window) return -1;
  glfwMakeContextCurrent    (window);
  glfwSetKeyCallback        (window, OnKey);
  glfwSetWindowSizeCallback (window, OnWindowResize);
  glfwSetWindowPos          (window, x, y);
  glfwSwapInterval          (swapInterval);
  glfwShowWindow            (window);

  // adjust initial window size
  OnWindowResize (window, w, h);

  // set material properties
  GLfloat mat_ambient[] = { 0.5f, 0.5f, 0.5f };
  GLfloat mat_diffuse[] = { 0.5f, 0.5f, 0.5f };
  GLfloat mat_specular[] = { 0.5f, 0.5f, 0.5f };
  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, mat_ambient);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, mat_diffuse);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, mat_specular);
  glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 1.0);

  // set light source
  GLfloat ambient[] = { 0.5f, 0.5f, 0.5f, 1.0f };
  GLfloat diffuse[] = { 0.8f, 0.8f, 0.8f, 1.0f };
  GLfloat specular[] = { 1.0f, 1.0f, 1.0f, 1.0f };
  glLightfv(GL_LIGHT0, GL_AMBIENT, ambient);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse);
  glLightfv(GL_LIGHT0, GL_SPECULAR, specular);
  glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 1.0);
  glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.0);
  glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 0.0);

  GLfloat lightPos[] = { 2.0, 0.0, 0.0, 1.0f };
  GLfloat lightDir[] = { -1.0, 0.0, 0.0, 1.0f };
  glLightfv(GL_LIGHT0, GL_POSITION, lightPos);
  glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, lightDir);
  glLightf(GL_LIGHT0, GL_SPOT_CUTOFF, 180);
  glLightf(GL_LIGHT0, GL_SPOT_EXPONENT, 1.0);
  glEnable(GL_LIGHTING);
  glEnable(GL_LIGHT0);

  return 0;
}



// haptic devices initialization
int
InitHaptics ()
{
  int count = dhdGetAvailableCount ();

  switch (count) {
  case 0:  printf ("no device found\n"); dhdSleep (2.0); return -1;
  case 1:  printf ("1 device available\n"); break;
  default: printf ("%d devices available\n", count); break;
  }

  if (count < 1) {
    printf ("error: %s\n", dhdErrorGetLastStr());
    return -1;
  }

  DeviceID        = new char[count];
  FingerCount     = new int[count];
  FingerPosGlobal = new Vector3d*[count];
  DeviceRotGlobal = new Matrix3d[count];
  HasRot          = new bool[count];
  HasGrip         = new bool[count];

  NumDevice = 0;
  for (int i=0; i<count; i++) {

    // open connection to device
    int id = dhdOpen ();
    if (id >= 0) {

      DeviceID[NumDevice] = id;

      printf ("device [%d]: %s\n", id, dhdGetSystemName());

      // configure simulation elements to match device capabilities
      HasRot[NumDevice]        = dhdHasWrist (NumDevice);
      HasGrip[NumDevice]       = dhdHasGripper (NumDevice);
      if (HasGrip[NumDevice])    FingerCount[NumDevice] = 2;
      else                       FingerCount[NumDevice] = 1;

      // allocate fingers
      FingerPosGlobal[NumDevice] = new Vector3d[FingerCount[NumDevice]];

      // increament actual devices count
      NumDevice++;
    }
    else printf ("error: cannot open device %d (%s)\n", i, dhdErrorGetLastStr());
  }

  if (NumDevice == 0) {
    printf ("no device detected\n");
    dhdSleep (2.0);
    exit (0);
  }

  printf ("\n");

  // register exit callback
  atexit (Close);

  return 0;
}



// simulation initialization
int
InitSimulation ()
{
  int i,d;

  glEnable (GL_DEPTH_TEST);
  glClearColor (0.0, 0.0, 0.0, 1.0);

  for (d=0; d<NumDevice; d++){
    for (i=0; i<FingerCount[d]; i++) {
      FingerPosGlobal[d][i].setZero();
    }
  }

  TorusPosGlobal.setZero();
  TorusRotGlobal.Identity();
  TorusRotGlobal = AngleAxisd (M_PI*45.0/180.0, Vector3d(0,1,-1));
  TorusVelGlobal.setZero();

  return 0;
}



int
main (int   argc,
      char *argv[])
{
  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  cout << endl;
  cout << "Force Dimension - OpenGL Integration Example " << major << "." << minor << "." << release << "." << revision << endl;
  cout << "Copyright (C) 2001-2020 Force Dimension" << endl;
  cout << "All Rights Reserved." << endl << endl;

  // initialize haptic devices
  if (InitHaptics () < 0) return -2;

  // initialize GLFW
  if (InitGLFW () < 0) return -1;

  // initialize simulation objects
  if (InitSimulation () < 0) return -3;

  // create a high priority haptic thread
#if defined(WIN32) || defined(WIN64)
  DWORD ThreadId;
  CreateThread (NULL, 0, (LPTHREAD_START_ROUTINE)(HapticsLoop), NULL, NULL, &ThreadId);
  SetThreadPriority(&ThreadId, THREAD_PRIORITY_ABOVE_NORMAL);
#else
  pthread_t handle;
  pthread_create (&handle, NULL, HapticsLoop, NULL);
  struct sched_param sp;
  memset (&sp, 0, sizeof(struct sched_param));
  sp.sched_priority = 10;
  pthread_setschedparam (handle, SCHED_RR, &sp);
#endif

  // display instructions
  printf ("\n");
  printf ("commands:\n\n");
  printf ("   'r' to toggle display of haptic rate\n");
  printf ("   'q' to quit\n");
  printf ("\n\n");

  // main graphic loop
  while (!glfwWindowShouldClose(window)) {

    // render graphics
    UpdateGraphics();

    // swap buffers
    glfwSwapBuffers(window);

    // process events
    glfwPollEvents();
  }

  // close window
  glfwDestroyWindow(window);

  // terminate GLFW library
  glfwTerminate();

  // exit
  return 0;
}
