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



// size of sphere
const double SphereRadius = 0.03;

// size of finger
const double FingerRadius = 0.005;

// status flags
bool SimulationOn;
bool SimulationFinished;

// device-specific globals
int      FingerCount;
Vector3d FingerPosGlobal[2];
bool     HasRot;
bool     HasGrip;

// position of sphere in global world coordinates
Vector3d SpherePosGlobal;

// text overlay globals
double LastTime;
double Freq;
char   Perf[50];
bool   ShowRate = true;

// object properties
const double Stiffness = 1000.0;

// GLFW display globals
GLFWwindow *window          = NULL;
int         width           = 0;
int         height          = 0;
int         swapInterval    = 1;



// OpenGL rendering
void
UpdateGraphics ()
{
  if (SimulationOn) {

    int i;

    cMatrixGL      mat;
    GLUquadricObj *sphere;
    double         deviceRotGlobal[3][3];

    // clean up
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // retrieve orientation frame (identity for 3-dof devices)
    dhdGetOrientationFrame (deviceRotGlobal);

    // render sphere
    glEnable  (GL_COLOR_MATERIAL);
    glColor3f (0.1f, 0.3f, 0.5f);
    mat.set (SpherePosGlobal);
    mat.glMatrixPushMultiply ();
    sphere = gluNewQuadric ();
    gluSphere (sphere, SphereRadius, 32, 32);
    mat.glMatrixPop ();

    // render finger(s)
    for (i=0; i<FingerCount; i++) {
      glColor3f (0.8f, 0.8f, 0.8f);
      mat.set (FingerPosGlobal[i], deviceRotGlobal);
      mat.glMatrixPushMultiply ();
      sphere = gluNewQuadric ();
      gluSphere (sphere, FingerRadius, 32, 32);

      // render a small frame
      if (HasRot) {
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

    // text overlay
    if (ShowRate) {
      if (dhdGetTime() - LastTime > 0.1) {
        Freq     = dhdGetComFreq();
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

  // close device
  dhdClose ();
}



// haptic thread
void*
HapticsLoop (void* pUserData)
{
  static double t0 = dhdGetTime ();
  double        t  = t0 + 0.001;

  int      i;
  Vector3d newFingerPosGlobal;
  Vector3d newFingerPosLocal;
  Vector3d forceGlobal[2];

  // start haptic simulation
  SimulationOn       = true;
  SimulationFinished = false;

  // start with no force
  forceGlobal[0].setZero ();
  forceGlobal[1].setZero ();

  // enable force
  dhdEnableForce (DHD_ON);

  // main haptic simulation loop
  while (SimulationOn) {

    t  = dhdGetTime ();
    t0 = t;

    double x, y, z;

    // adapt behavior to device capabilities
    if (HasGrip) {
      dhdGetGripperThumbPos (&x, &y, &z);
      FingerPosGlobal[0] << x, y, z;

      dhdGetGripperFingerPos (&x, &y, &z);
      FingerPosGlobal[1] << x, y, z;
    }
    else {
      dhdGetPosition (&x, &y, &z);
      FingerPosGlobal[0] << x, y, z;
    }

    // compute interaction between sphere and each finger
    for (i=0; i<FingerCount; i++) {

      // compute penetration
      Vector3d dir  = (FingerPosGlobal[i] - SpherePosGlobal).normalized();
      double   dist = (FingerPosGlobal[i] - SpherePosGlobal).norm() - SphereRadius - FingerRadius;

      // compute force
      if (dist < 0.0) forceGlobal[i] = -dist * Stiffness * dir;
      else            forceGlobal[i].setZero ();
    }

    // compute projected force on each gripper finger
    Vector3d force;
    double   gripperForce = 0.0;
    if (HasGrip) {

      // compute total force
      force = forceGlobal[0] + forceGlobal[1];
      Vector3d gripdir = FingerPosGlobal[1] - FingerPosGlobal[0];

      // if force is not null
      if (gripdir.norm() > 0.00001) {

        // project force on mobile gripper finger (forceGlobal[1]) onto gripper opening vector (gripdir)
        gripdir.normalize ();
        Vector3d gripper = (forceGlobal[1].dot (gripdir) / (gripdir.squaredNorm())) * gripdir;
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
    else force = forceGlobal[0];

    // apply all forces at once
    dhdSetForceAndGripperForce (force(0), force(1), force(2), gripperForce);
  }

  // close connection with haptic device
  dhdClose ();

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
  if ((key == GLFW_KEY_ESCAPE) || (key == GLFW_KEY_Q)) {
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
  window = glfwCreateWindow(w, h, "Force Dimension - Sphere Example", NULL, NULL);
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
  if (dhdOpen () >= 0) {
    printf ("%s device detected\n", dhdGetSystemName());

    // set device capabilities
    FingerCount = 1;
    HasRot      = dhdHasWrist ();
    HasGrip     = dhdHasGripper ();

    if (HasGrip) FingerCount = 2;
    else         FingerCount = 1;
  }

  else {
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
  int i;

  glEnable (GL_DEPTH_TEST);
  glClearColor (0.0, 0.0, 0.0, 1.0);

  for (i=0; i<FingerCount; i++) FingerPosGlobal[i].setZero();

  SpherePosGlobal.setZero();

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
  cout << "Force Dimension - OpenGL Sphere Example " << major << "." << minor << "." << release << "." << revision << endl;
  cout << "Copyright (C) 2001-2020 Force Dimension" << endl;
  cout << "All Rights Reserved." << endl << endl;

  // initialize haptic devices
  InitHaptics ();

  // initialize GLFW
  InitGLFW ();

  // initialize simulation objects
  InitSimulation ();

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
