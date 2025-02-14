///////////////////////////////////////////////////////////////////////////////
//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
//  Version 3.9.1
///////////////////////////////////////////////////////////////////////////////



///////////////////////////////////////////////////////////////////////////////
// INCLUDE FILES
///////////////////////////////////////////////////////////////////////////////

// General
#include <stdio.h>
#include <string.h>
#include <iostream>
using namespace std;

// CHAI3D geometry and math library
#include "CGeometry.h"
using namespace chai3d;

// Force Dimension SDK
#include "dhdc.h"

// GLFW library
#include <GLFW/glfw3.h>
#ifdef MACOSX
#include "OpenGL/glu.h"
#else
#include "GL/glu.h"
#endif

/* 
 *  Acknowledgement:
 *  the following font data structure, font typefaces and font rendering implementation
 *  is adapted from the freeglut project implementation of glutRenderBitmap().
 *  Credits to http://freeglut.sourceforge.net 
 */

// font data structure
struct font_struct
{
  char*           name;
  int             quantity;
  int             height;
  const GLubyte** characters;
  float           xorig, yorig;
};

// import font data
#include "renderBitmapCharacterGL.dat"
#define HELVETICA12 0
font_struct *font_list [] = { &FontHelvetica12 };

// font rendering function
void renderBitmapCharacter (int character, int font_id)
{
  const font_struct *font = font_list[font_id];
  const GLubyte     *face = font->characters[character];

  glPushClientAttrib (GL_CLIENT_PIXEL_STORE_BIT);
  glPixelStorei      (GL_UNPACK_SWAP_BYTES,  GL_FALSE);
  glPixelStorei      (GL_UNPACK_LSB_FIRST,   GL_FALSE);
  glPixelStorei      (GL_UNPACK_ROW_LENGTH,  0);
  glPixelStorei      (GL_UNPACK_SKIP_ROWS,   0);
  glPixelStorei      (GL_UNPACK_SKIP_PIXELS, 0);
  glPixelStorei      (GL_UNPACK_ALIGNMENT,   1);
  glBitmap(face[0], font->height, font->xorig, font->yorig,(float)(face[0]), 0.0,(face+1));
  glPopClientAttrib( );
}



///////////////////////////////////////////////////////////////////////////////
// GLOBAL VARIABLES
///////////////////////////////////////////////////////////////////////////////

// status flags
bool SimulationOn;
bool SimulationFinished;

// haptic device
cVector3d DevicePos;
cMatrix3d DeviceRot;
double    DeviceGripperAngleDeg;
cVector3d DeviceLinVel;
cVector3d DeviceAngVel;
uint      DeviceSwitches;
cVector3d DeviceForce;
cVector3d DeviceTorque;
double    DeviceGripperForce;
double    DeviceRollAngleRad;

// virtual frame
cVector3d FramePos;
cMatrix3d FrameRot;
cTransform Device_T_Frame;
bool FlagMoveFrame = false;

// hold position constraint
cVector3d HoldPos;
cMatrix3d HoldRot;
bool FlagHoldPosition = false;;
bool FlagHoldPositionReady = false;

// text overlay globals
double LastTime;
double Freq;
char   Perf[50];

// GLFW display globals
GLFWwindow *window          = NULL;
int         width           = 0;
int         height          = 0;
int         swapInterval    = 1;



// OpenGL rendering
void
UpdateGraphics ()
{
  ///////////////////////////////////////////////////////////////////////////
  // INITIALIZE RENDERING
  ///////////////////////////////////////////////////////////////////////////

  GLUquadricObj *sphere;
  cTransform     mat;

  // initialize settings
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glEnable(GL_DEPTH_TEST);
  glClearColor(0.0, 0.0, 0.0, 1.0);

  ///////////////////////////////////////////////////////////////////////////
  // DISPLAY HAPTIC DEVICE
  ///////////////////////////////////////////////////////////////////////////

  // display haptic device
  mat.set(DevicePos, DeviceRot);
  glPushMatrix();
  glMultMatrixd((const double *)mat.getData());

  // render sphere
  glEnable(GL_COLOR_MATERIAL);
  glColor3f(0.5f, 0.5f, 0.5f);
  sphere = gluNewQuadric();
  gluSphere(sphere, 0.005, 32, 32);

  // render frame
  glDisable(GL_LIGHTING);
  glBegin(GL_LINES);
  glColor3f(1.0f, 0.0f, 0.0f);
  glVertex3d(0.00, 0.00, 0.00);
  glVertex3d(0.02, 0.00, 0.00);
  glColor3f(0.0f, 1.0f, 0.0f);
  glVertex3d(0.00, 0.00, 0.00);
  glVertex3d(0.00, 0.02, 0.00);
  glColor3f(0.0f, 0.0f, 1.0f);
  glVertex3d(0.00, 0.00, 0.00);
  glVertex3d(0.00, 0.00, 0.02);
  glEnd();
  glEnable(GL_LIGHTING);

  // render gripper
  glDisable(GL_LIGHTING);
  glBegin(GL_LINE_STRIP);
  glColor3f(1.0f, 1.0f, 1.0f);
  glVertex3d(0.0, 0.0, 0.0);
  for (int i = -10; i <= 10; i++)
  {
    double angle = 0.1 * (double)(i)* DeviceGripperAngleDeg;
    double px = 0.1 * cCosDeg(angle);
    double py = 0.1 * cSinDeg(angle);
    glVertex3d(-px, py, 0.0);
  }
  glVertex3d(0.0, 0.0, 0.0);
  glColor3f(0.5f, 0.5f, 0.5f);
  glVertex3d(-0.1, 0.0, 0.0);
  glEnd();
  glEnable(GL_LIGHTING);
  glPopMatrix();


  ///////////////////////////////////////////////////////////////////////////
  // DISPLAY FRAME
  ///////////////////////////////////////////////////////////////////////////

  // display virtual object frame
  mat.set(FramePos, FrameRot);
  glPushMatrix();
  glMultMatrixd((const double *)mat.getData());

  // render sphere
  glEnable(GL_COLOR_MATERIAL);
  glColor3f(0.5f, 0.5f, 0.5f);
  sphere = gluNewQuadric();
  gluSphere(sphere, 0.01, 32, 32);

  // render frame
  glDisable(GL_LIGHTING);
  glBegin(GL_LINES);
  glColor3f(1.0f, 0.0f, 0.0f);
  glVertex3d(0.00, 0.00, 0.00);
  glVertex3d(0.04, 0.00, 0.00);
  glColor3f(0.0f, 1.0f, 0.0f);
  glVertex3d(0.00, 0.00, 0.00);
  glVertex3d(0.00, 0.04, 0.00);
  glColor3f(0.0f, 0.0f, 1.0f);
  glVertex3d(0.00, 0.00, 0.00);
  glVertex3d(0.00, 0.00, 0.04);
  glEnd();
  glEnable(GL_LIGHTING);

  glPopMatrix();


  ///////////////////////////////////////////////////////////////////////////
  // DISPLAY LINE BETWEEN FRAME AND DEVICE
  ///////////////////////////////////////////////////////////////////////////

  glDisable(GL_LIGHTING);
  glBegin(GL_LINES);
  glColor3f(0.5f, 0.5f, 0.5f);
  glVertex3d(DevicePos(0), DevicePos(1), DevicePos(2));
  glVertex3d(FramePos(0), FramePos(1), FramePos(2));
  glEnd();
  glEnable(GL_LIGHTING);

  ///////////////////////////////////////////////////////////////////////////
  // DISPLAY REFRESH RATE
  ///////////////////////////////////////////////////////////////////////////

  // render refresh rate text overlay
  if (dhdGetTime() - LastTime > 0.1)
  {
    Freq = dhdGetComFreq();
    LastTime = dhdGetTime();
    sprintf(Perf, "%0.03f kHz", Freq);
  }

  glDisable(GL_LIGHTING);
  glColor3f(1.0, 1.0, 1.0);
  glRasterPos3f(0.0f, -0.01f, -0.1f);
  for (char *c = Perf; *c != '\0'; c++) renderBitmapCharacter(*c, HELVETICA12);
  glEnable(GL_LIGHTING);


  ///////////////////////////////////////////////////////////////////////////
  // DISPLAY STATUS OF SWITCHES
  ///////////////////////////////////////////////////////////////////////////

  // render status of switches
  glDisable(GL_LIGHTING);
  glColor3f(1.0, 1.0, 1.0);
  glRasterPos3f(0.0f, -0.03f, 0.1f);
  for (int i = 0; i < 16; i++)
  {
    // display bit status
    if (cCheckBit(DeviceSwitches, i)) renderBitmapCharacter('1', HELVETICA12);
    else                              renderBitmapCharacter('0', HELVETICA12);

    // display space
    renderBitmapCharacter(' ', HELVETICA12);
  }
  glEnable(GL_LIGHTING);


  ///////////////////////////////////////////////////////////////////////////
  // FINALIZE RENDERING
  ///////////////////////////////////////////////////////////////////////////

  // check for any OpenGL errors
  GLenum err = glGetError();
  if (err != GL_NO_ERROR) printf("error:  %s\n", gluErrorString(err));
}



// exit callback
void
Close(void)
{
  // finish haptic loop
  SimulationOn = false;
  while (!SimulationFinished) dhdSleep(0.1);

  // close device
  dhdClose();
}



// haptic thread
void*
HapticsLoop(void* pUserData)
{
  static double t0 = dhdGetTime();
  double        t = t0 + 0.001;

  // start haptic simulation
  SimulationOn = true;
  SimulationFinished = false;

  // enable force
  dhdEnableForce(DHD_ON);

  // enable virtual switch
  dhdEmulateButton(DHD_ON);

  // main haptic simulation loop
  while (SimulationOn) {

    ///////////////////////////////////////////////////////////////////////
    // UPDATE TIME
    ///////////////////////////////////////////////////////////////////////

    // read time
    t = dhdGetTime();
    t0 = t;


    ///////////////////////////////////////////////////////////////////////
    // RETRIEVE DATA FROM HAPTIC DEVICE
    ///////////////////////////////////////////////////////////////////////

    // read position of haptic device
    double posX, posY, posZ;
    dhdGetPosition(&posX, &posY, &posZ);
    DevicePos.set(posX, posY, posZ);

    // read orientation of haptic device
    double r[3][3];
    dhdGetOrientationFrame(r);
    DeviceRot.set(r[0][0], r[0][1], r[0][2],
        r[1][0], r[1][1], r[1][2],
        r[2][0], r[2][1], r[2][2]);

    // read gripper angle of haptic device
    dhdGetGripperAngleDeg(&DeviceGripperAngleDeg);

    // read linear velocity of haptic device
    double linVelX, linVelY, linVelZ;
    dhdGetLinearVelocity(&linVelX, &linVelY, &linVelZ);
    DeviceLinVel.set(linVelX, linVelY, linVelZ);

    // read angular velocity of haptic device
    double angVelX, angVelY, angVelZ;
    dhdGetAngularVelocityRad(&angVelX, &angVelY, &angVelZ);
    DeviceAngVel.set(angVelX, angVelY, angVelZ);

    // read user switches and data
    DeviceSwitches = dhdGetButtonMask();

    ///////////////////////////////////////////////////////////////////////
    // MOVE FRAME
    ///////////////////////////////////////////////////////////////////////

    if (cCheckBit(DeviceSwitches, 0))
    {
      if (FlagMoveFrame)
      {
        // compute the tranformation matrix from World to Device
        cTransform World_T_Device;
        World_T_Device.set(DevicePos, DeviceRot);

        // compute the new position of the Frame
        cTransform World_T_Frame = World_T_Device * Device_T_Frame;

        // apply values to frame
        FramePos = World_T_Frame.getLocalPos();
        FrameRot = World_T_Frame.getLocalRot();
      }
      else
      {
        // compute the tranformation matrix from World to Device
        cTransform World_T_Device;
        World_T_Device.set(DevicePos, DeviceRot);

        // compute the tranformation matrix from World to Frame
        cTransform World_T_Frame;
        World_T_Frame.set(FramePos, FrameRot);

        // compute the tranformation matrix from Device to World
        cTransform Device_T_World = World_T_Device;
        Device_T_World.invert();

        // compute the transformation matrix from Device to Frame;
        Device_T_Frame = Device_T_World * World_T_Frame;

        // enable frame motion
        FlagMoveFrame = true;
      }
    }
    else
    {
      // enable frame motion
      FlagMoveFrame = false;
    }


    ///////////////////////////////////////////////////////////////////////
    // COMPUTE INTERACTION FORCES
    ///////////////////////////////////////////////////////////////////////

    cVector3d deviceForce(0.0, 0.0, 0.0);
    cVector3d deviceTorque(0.0, 0.0, 0.0);
    double deviceGripperForce = 0.0;

    // define stiffness and viscosity used to constrain the device position
    const double Kp = 2000.0;  // N/m
    const double Kv =   10.0;  // N/(m/s)

    // compute force to hold device in position
    if (FlagHoldPosition)
    {
      if (FlagHoldPositionReady)
      {
        // compute reaction force
        cVector3d force = -Kp * (DevicePos - HoldPos) - Kv * DeviceLinVel;

        // compute reaction torque
        const double Kpr = 5.0;
        const double Kvr = 0.05;
        cVector3d axis;
        double angle;
        cMatrix3d deltaRotation = cTranspose(DeviceRot) * HoldRot;
        deltaRotation.toAxisAngle(axis, angle);
        cVector3d torque = DeviceRot * ((Kpr * angle) * axis) - Kvr * DeviceAngVel;

        // add all forces together
        deviceForce = deviceForce + force;
        deviceTorque = deviceTorque + torque;
      }
      else
      {
        HoldPos = DevicePos;
        HoldRot = DeviceRot;
        FlagHoldPositionReady = true;
      }
    }


    ///////////////////////////////////////////////////////////////////////
    // SEND FORCE AND TORQUE TO HAPTIC DEVICE
    ///////////////////////////////////////////////////////////////////////

    // update global variables
    DeviceForce = deviceForce;
    DeviceTorque = deviceTorque;
    DeviceGripperForce = deviceGripperForce;

    // limit torque to maximum value
    double MAX_TORQUE = 0.3;
    if (DeviceTorque.length() > MAX_TORQUE)
    {
      DeviceTorque = MAX_TORQUE * cNormalize(DeviceTorque);
    }

    // apply forces and torques at once
    dhdSetForceAndTorqueAndGripperForce(DeviceForce(0), DeviceForce(1), DeviceForce(2),
        DeviceTorque(0), DeviceTorque(1), DeviceTorque(2),
        DeviceGripperForce);
  }

  // close connection with haptic device
  dhdClose();

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
    while (!SimulationFinished) dhdSleep(0.01);
    exit(0);
  }

  // toggle hold device in position
  if (key == GLFW_KEY_H) {
    FlagHoldPosition = !FlagHoldPosition;
    FlagHoldPositionReady = false;
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
  window = glfwCreateWindow(w, h, "Force Dimension - Transformations Example", NULL, NULL);
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
InitHaptics()
{
  if (dhdOpen() >= 0) {
    printf("%s device detected\n", dhdGetSystemName());
  }

  else {
    printf("no device detected\n");
    dhdSleep(2.0);
    exit(0);
  }

  printf("\n");

  // register exit callback
  atexit(Close);

  return 0;
}



// simulation initialization
int
InitSimulation()
{
  FramePos.set(0.0, 0.0, 0.0);
  FrameRot.identity();

  return 0;
}



int
main(int   argc,
     char *argv[])
{
  // message
  int major, minor, release, revision;
  dhdGetSDKVersion(&major, &minor, &release, &revision);
  cout << endl;
  cout << "Force Dimension - OpenGL Transformations Example " << major << "." << minor << "." << release << "." << revision << endl;
  cout << "Copyright (C) 2001-2020 Force Dimension" << endl;
  cout << "All Rights Reserved." << endl << endl;

  // initialize haptic devices
  InitHaptics();

  // initialize GLFW
  InitGLFW ();

  // initialize simulation objects
  InitSimulation();

  // create a high priority haptic thread
#if defined(WIN32) || defined(WIN64)
  DWORD ThreadId;
  CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE)(HapticsLoop), NULL, NULL, &ThreadId);
  SetThreadPriority(&ThreadId, THREAD_PRIORITY_ABOVE_NORMAL);
#else
  pthread_t handle;
  pthread_create(&handle, NULL, HapticsLoop, NULL);
  struct sched_param sp;
  memset(&sp, 0, sizeof(struct sched_param));
  sp.sched_priority = 10;
  pthread_setschedparam(handle, SCHED_RR, &sp);
#endif

  // display instructions
  printf ("\n");
  printf ("commands:\n\n");
  printf ("   'h' to toggle hold device in position\n");
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
