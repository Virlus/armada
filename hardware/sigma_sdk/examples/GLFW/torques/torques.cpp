//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1



#include <stdio.h>
#include <string.h>
#include <iostream>
using namespace std;

#include "CMacrosGL.h"
using namespace Eigen;

#include "dhdc.h"

// GLFW library
#include <GLFW/glfw3.h>



// object stiffness
double Stiffness   = 2012;
double TorqueGain  =    1.0;

// set size of cube (half edge)
const float CubeSize = 0.06f;

// status flags
bool SimulationOn;
bool SimulationFinished;
int  Status[DHD_MAX_STATUS];
bool HapticsON = false;
bool TorquesON = true;
bool ForcesON  = true;

// text overlay globals
double LastTime;
double Freq;
char   Perf[50];
bool   ShowRate = true;

// white diffuse light
GLfloat LightAmbient[]  = {0.5f, 0.5f, 0.5f, 1.0f};
GLfloat LightDiffuse[]  = {0.8f, 0.8f, 0.8f, 1.0f};
GLfloat LightSpecular[] = {1.0f, 1.0f, 1.0f, 1.0f};

// light source position
GLfloat LightPosition[] = {1.0f, 0.5f, 0.8f, 0.0f};

// normals for the 6 faces of a cube
GLfloat N[6][3]   = { {-1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 0.0, 0.0},
                      {0.0, -1.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, -1.0} };

// vertex indices for the 6 faces of a cube
GLint Faces[6][4] = { {0, 1, 2, 3}, {3, 2, 6, 7}, {7, 6, 5, 4},
                      {4, 5, 1, 0}, {5, 6, 2, 1}, {7, 4, 0, 3} };

// will be filled in with X,Y,Z vertices
GLfloat V[8][3];

// GLFW display globals
GLFWwindow *window          = NULL;
int         width           = 0;
int         height          = 0;
int         swapInterval    = 1;



// OpenGL rendering
void
UpdateGraphics (void)
{
  if (SimulationOn) {

    int            i;
    double         posX, posY, posZ;
    cMatrixGL      mat;
    GLUquadricObj *sphere;
    double         deviceRot[3][3];
    Vector3d       devicePos;

    // clean up
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // get info from device
    dhdGetPosition (&posX, &posY, &posZ);
    devicePos << posX, posY, posZ;
    dhdGetOrientationFrame (deviceRot);

    // render cube
    mat.set (devicePos, deviceRot);
    mat.glMatrixPushMultiply ();

    glEnable  (GL_COLOR_MATERIAL);
    glColor3f (0.1f, 0.3f, 0.5f);

    for (i=0; i<6; i++) {
      glBegin     (GL_QUADS);
      glNormal3fv (&N[i][0]);
      glVertex3fv (&V[Faces[i][0]][0]);
      glVertex3fv (&V[Faces[i][1]][0]);
      glVertex3fv (&V[Faces[i][2]][0]);
      glVertex3fv (&V[Faces[i][3]][0]);
      glEnd       ();
    }

    mat.glMatrixPop ();

    // render sphere at center of workspace
    glColor3f (0.8f, 0.8f, 0.8f);
    sphere = gluNewQuadric ();
    gluSphere (sphere, 0.005, 64, 64);

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
    if (err != GL_NO_ERROR) printf ("error: %s\n", gluErrorString(err));
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



// collision model
void
ComputeInteractions (Vector3d& pos,
                     Matrix3d& rot,
                     Vector3d& force,
                     Vector3d& torque)
{
  static const float size = CubeSize / 2.0f;

  Matrix3d localRotTrans;
  Vector3d localPos;
  Vector3d localForce  (0, 0, 0);
  Vector3d localTorque (0, 0, 0);

  // compute position of device in locale coordinates of cube
  localRotTrans =   rot.transpose ();
  localPos      = - localRotTrans * pos;

  // compute interaction force and torque
  if ((localPos(0) < size) && (localPos(0) > -size) &&
      (localPos(1) < size) && (localPos(1) > -size) &&
      (localPos(2) < size) && (localPos(2) > -size))
  {
    double   depth = size;
    double   t_depth;
    Vector3d normal (0,0,1);
    Vector3d nForce;

    // check all size walls of cube
    t_depth = fabs( size - localPos(0)); if (t_depth < depth) { depth = t_depth; normal <<  1, 0, 0; }
    t_depth = fabs(-size - localPos(0)); if (t_depth < depth) { depth = t_depth; normal << -1, 0, 0; }
    t_depth = fabs( size - localPos(1)); if (t_depth < depth) { depth = t_depth; normal <<  0, 1, 0; }
    t_depth = fabs(-size - localPos(1)); if (t_depth < depth) { depth = t_depth; normal <<  0,-1, 0; }
    t_depth = fabs( size - localPos(2)); if (t_depth < depth) { depth = t_depth; normal <<  0, 0, 1; }
    t_depth = fabs(-size - localPos(2)); if (t_depth < depth) { depth = t_depth; normal <<  0, 0,-1; }

    // compute reaction force
    localForce  = -depth * Stiffness * normal;
    nForce      = - localForce;
    localTorque = -TorqueGain * localPos.cross (nForce);
  }

  // convert results in global coordinates
  force  = rot * localForce;
  torque = rot * localTorque;
}



// haptic thread
void*
HapticsLoop (void* pUserData)
{
  double   posX, posY, posZ;
  Vector3d deviceForce;
  Vector3d deviceTorque;
  Vector3d devicePos;
  double   r[3][3];
  Matrix3d deviceRot;
  bool     force;
  bool     btnDown;

  // start haptic simulation
  force              = false;
  btnDown            = false;
  SimulationOn       = true;
  SimulationFinished = false;

  // enable force
  dhdEnableForce (DHD_ON);

  // main haptic simulation loop
  while (SimulationOn) {

    // init variables
    posX = posY = posZ = 0.0;

    // read position of haptic device
    dhdGetPosition (&posX, &posY, &posZ);
    devicePos << posX,  posY,  posZ;

    // read orientation of haptic device
    dhdGetOrientationFrame (r);
    deviceRot << r[0][0], r[0][1], r[0][2],
                 r[1][0], r[1][1], r[1][2],
                 r[2][0], r[2][1], r[2][2];

    // compute forces and torques
    deviceForce.setZero ();
    deviceTorque.setZero ();
    ComputeInteractions (devicePos, deviceRot, deviceForce, deviceTorque);

    // only enable forces once the device if in free space
    if (!HapticsON) {
      if (deviceForce.norm() == 0) {
        HapticsON = true;
      }
      else {
        deviceForce.setZero();
        deviceTorque.setZero();
      }
    }

    // disable torques if required
    if (!TorquesON) deviceTorque.setZero();

    // send forces to device
    dhdSetForceAndTorqueAndGripperForce (deviceForce(0),  deviceForce(1),  deviceForce(2),
                                         deviceTorque(0), deviceTorque(1), deviceTorque(2),
                                         0.0);
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

  // toggle torques
  if (key == GLFW_KEY_T) TorquesON = !TorquesON;

  // toggle forces
  if (key == GLFW_KEY_F) {
    ForcesON = !ForcesON;
    if (ForcesON) dhdEnableForce (DHD_ON);
    else          dhdEnableForce (DHD_OFF);
  }
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
  window = glfwCreateWindow(w, h, "Force Dimension - Torques Example", NULL, NULL);
  if (!window) return -1;
  glfwMakeContextCurrent    (window);
  glfwSetKeyCallback        (window, OnKey);
  glfwSetWindowSizeCallback (window, OnWindowResize);
  glfwSetWindowPos          (window, x, y);
  glfwSwapInterval          (swapInterval);
  glfwShowWindow            (window);

  // adjust initial window size
  OnWindowResize (window, w, h);

  // compute size of half edge of cube
  static const float size = CubeSize / 2.0f;

  // setup cube vertex data
  V[0][0] = V[1][0] = V[2][0] = V[3][0] = -size;
  V[4][0] = V[5][0] = V[6][0] = V[7][0] =  size;
  V[0][1] = V[1][1] = V[4][1] = V[5][1] = -size;
  V[2][1] = V[3][1] = V[6][1] = V[7][1] =  size;
  V[0][2] = V[3][2] = V[4][2] = V[7][2] =  size;
  V[1][2] = V[2][2] = V[5][2] = V[6][2] = -size;

  // enable a single OpenGL light.
  glLightfv (GL_LIGHT0, GL_AMBIENT,  LightAmbient);
  glLightfv (GL_LIGHT0, GL_DIFFUSE,  LightDiffuse);
  glLightfv (GL_LIGHT0, GL_SPECULAR, LightSpecular);
  glLightfv (GL_LIGHT0, GL_POSITION, LightPosition);
  glEnable  (GL_LIGHT0);
  glEnable  (GL_LIGHTING);

  // use depth buffering for hidden surface elimination.
  glEnable (GL_DEPTH_TEST);

  return 0;
}



// haptic devices initialization
int
InitHaptics ()
{
  // open the first available device
  if (dhdOpen () < 0) {
    printf ("error: cannot open device (%s)\n", dhdErrorGetLastStr());
    dhdSleep (2.0);
    return -1;
  }

  // identify device
  printf ("%s device detected\n\n", dhdGetSystemName());

  // register exit callback
  atexit (Close);

  // adjust stiffness for some devices
  switch (dhdGetSystemType ()) {
  case DHD_DEVICE_SIGMA331:
  case DHD_DEVICE_SIGMA331_LEFT:
    Stiffness    = 6000;
    TorqueGain  =     1.0;
    break;
  }

  return 0;
}



// entry point
int
main (int    argc,
      char **argv)
{
  // message
  int major, minor, release, revision;
  dhdGetSDKVersion (&major, &minor, &release, &revision);
  cout << endl;
  cout << "Force Dimension - OpenGL Torques Example " << major << "." << minor << "." << release << "." << revision << endl;
  cout << "Copyright (C) 2001-2020 Force Dimension" << endl;
  cout << "All Rights Reserved." << endl << endl;

  // initialize haptic devices
  InitHaptics ();

  // initialize GLFW
  InitGLFW ();

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
  printf ("   't' to enable or disable torques on wrist actuated devices\n");
  printf ("   'f' to enable or disable forces\n");
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

