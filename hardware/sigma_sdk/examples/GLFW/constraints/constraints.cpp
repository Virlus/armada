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

// GLU library
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

// haptic device status
bool FlagSaturation         = false;

// hold position constraint
cVector3d HoldPos;
bool FlagHoldPosition       = false;;
bool FlagHoldPositionReady  = false;

// workspace constraint
double WorkspaceRadius      = 0.05;
bool FlagWorkspace          = true;
bool FlagWorkspaceReady     = false;

// cone workspace
double    ConeAngle         = cDegToRad(10.0);
double    RollAngle         = cDegToRad(40.0);
cVector3d ConeAxis;
cVector3d RollAxis;
cMatrix3d ConeRot;
bool      FlagCone          = false;
bool      FlagConeReady     = false;

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

  /*
  if (FlagSaturation)
    glColor3f(0.5f, 0.0f, 0.0f);
  else
    glColor3f(0.5f, 0.5f, 0.5f);
  */
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
  for (int i=-10; i<=10; i++)
  {
    double angle = 0.1 * (double)(i) * DeviceGripperAngleDeg;
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
  // DISPLAY CONE
  ///////////////////////////////////////////////////////////////////////////

  // render cone and roll constraint
  if (FlagConeReady)
  {
    // render cone
    mat.set(DevicePos, ConeRot);
    glPushMatrix();
    glMultMatrixd((const double *)mat.getData());

    // render cone axis
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES);
    glColor3f(0.0f, 1.0f, 1.0f);
    glVertex3d(0.0, 0.0, 0.0);
    glVertex3d(-0.1, 0.0, 0.0);
    glEnd();

    // render gripper fan
    glBegin(GL_LINE_STRIP);
    for (int i = 0; i <= 36; i++)
    {
      double SIZE = 0.1;
      double radius = SIZE * cTanRad(ConeAngle);
      double angle = 10 * (double)(i);
      double px = radius * cCosDeg(angle);
      double py = radius * cSinDeg(angle);
      glVertex3d(-SIZE, px, py);
    }
    glEnd();
    glEnable(GL_LIGHTING);
    glPopMatrix();

    // render roll disk and limits
    mat.set(DevicePos, DeviceRot);
    glPushMatrix();
    glMultMatrixd((const double *)mat.getData());

    // render lines and polygons in two passes
    for (int pass = 0; pass < 2; pass++)
    {
      if (pass == 0)
      {
        glColor3f(1.0f, 1.0f, 1.0f);
        glDepthMask(GL_TRUE);
        glBegin(GL_LINE_STRIP);
      }
      else if (pass == 1)
      {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glColor4f(1.0f, 1.0f, 1.0f, 0.3f);
        glDepthMask(GL_FALSE);
        glBegin(GL_TRIANGLE_FAN);
      }

      glVertex3d(0.0, 0.0, 0.0);
      double angleMin = -RollAngle + DeviceRollAngleRad;
      double angleMax = RollAngle + DeviceRollAngleRad;
      for (int i = 0; i <= 36; i++)
      {
        double angle = angleMax + ((C_TWO_PI - (angleMax - angleMin)) / 36.0) * (double)(i);
        double px = 0.02 * cCosRad(angle);
        double py = 0.02 * cSinRad(angle);
        glVertex3d(0, -py, px);
      }
      glVertex3d(0.0, 0.0, 0.0);
      glEnd();
      glDisable(GL_BLEND);
      glDepthMask(GL_TRUE);
    }

    glEnable(GL_LIGHTING);
    glPopMatrix();
  }


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
    if (cCheckBit(DeviceSwitches, i))
      renderBitmapCharacter('1', HELVETICA12);
    else
      renderBitmapCharacter('0', HELVETICA12);

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

        // add all forces together
        deviceForce = deviceForce + force;
      }
      else
      {
        HoldPos = DevicePos;
        FlagHoldPositionReady = true;
      }
    }

    // compute workspace constraint
    if (FlagWorkspace)
    {
      double distance = DevicePos.length();
      if (FlagWorkspaceReady)
      {
        double depth = distance - WorkspaceRadius;
        if (depth > 0.0)
        {
          // compute surface normal vector
          cVector3d n = -cNormalize(DevicePos);

          // compute reaction force
          cVector3d force = Kp * depth * n;

          // compute damping term parallel to normal
          cVector3d damping = -Kv * cProject(DeviceLinVel, n);

          // add all forces together
          deviceForce = deviceForce + (force + damping);
        }
      }
      else
      {
        if (distance < WorkspaceRadius)
          FlagWorkspaceReady = true;
      }
    }

    // compute cone constraint
    if (FlagCone)
    {
      if (FlagConeReady)
      {
        ///////////////////////////////////////////////////////////////
        // CONE CONSTRAINT
        ///////////////////////////////////////////////////////////////

        // compute tool axis and angle
        cVector3d toolAxis = -DeviceRot.getCol0();
        double toolAngle = cAngle(toolAxis, ConeAxis);

        // compute constraint
        if (toolAngle > ConeAngle)
        {
          // compute cross product between cone axis and direction
          // of haptic device; this gives us the torque direction.
          cVector3d t = cNormalize(cCross(ConeAxis, toolAxis));

          // compute angle difference
          double deltaAngle = toolAngle - ConeAngle;

          // compute torque
          cVector3d torque = -10.0 * deltaAngle * t;

          // compute damping
          cVector3d damping = -0.02 * cProject(DeviceAngVel, t);

          // add torque from cone constraint
          deviceTorque = deviceTorque + (torque + damping);
        }

        ///////////////////////////////////////////////////////////////
        // ROLL CONSTRAINT
        ///////////////////////////////////////////////////////////////

        // compute roll axis and angle
        cVector3d toolRoll = DeviceRot.getCol2();
        cVector3d ProjectedRollAxis = cProjectPointOnPlane(RollAxis, cVector3d(0,0,0), toolAxis);
        double rollAngle = cAngle(toolRoll, ProjectedRollAxis);

        // compute cross product between cone axis and direction
        // of haptic device; this gives us the torque direction.
        cVector3d t = cNormalize(ConeAxis);

        // compute torque sign
        double sign = 1.0;
        cVector3d c = cCross(toolRoll, ProjectedRollAxis);
        if (cAngle(t, c) < C_PI_DIV_2)
        {
          sign = -1.0;
        }

        // store value
        DeviceRollAngleRad = sign * rollAngle;

        // compute constraint
        if (rollAngle > RollAngle)
        {
          // compute angle difference
          double deltaAngle = rollAngle - RollAngle;

          // compute torque
          cVector3d torque = -4.0 * sign * deltaAngle * t;

          // compute damping
          cVector3d damping = -0.02 * cProject(DeviceAngVel, t);

          // add torque from cone constraint
          deviceTorque = deviceTorque + (torque + damping);
        }
      }
      else
      {
        ConeAxis = -DeviceRot.getCol0();
        RollAxis = DeviceRot.getCol2();
        ConeRot = DeviceRot;
        FlagConeReady = true;
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
    int error = dhdSetForceAndTorqueAndGripperForce(DeviceForce(0), DeviceForce(1), DeviceForce(2),
        DeviceTorque(0), DeviceTorque(1), DeviceTorque(2),
        DeviceGripperForce);

    // check for if any force saturation occured
    if (error == 2) { FlagSaturation = true; }
    else { FlagSaturation = false; }
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
  if (action == GLFW_RELEASE) return;

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

  // toggle workspace constraint
  if (key == GLFW_KEY_W) {
    FlagWorkspace = !FlagWorkspace;
    FlagWorkspaceReady = false;
  }

  // toggle cone constraint
  if (key == GLFW_KEY_C) {
    FlagCone = !FlagCone;
    FlagConeReady = false;
  }

  // reduce cone angle
  if (key == GLFW_KEY_1) {
    ConeAngle = cMax(ConeAngle - cDegToRad(1.0), cDegToRad(0.0));
  }

  // increase cone angle
  if (key == GLFW_KEY_2) {
    ConeAngle = cMin(ConeAngle + cDegToRad(1.0), cDegToRad(35.0));
  }

  // reduce roll angle
  if (key == GLFW_KEY_3) {
    RollAngle = cMax(RollAngle - cDegToRad(1.0), cDegToRad(0.0));
  }

  // increase roll angle
  if (key == GLFW_KEY_4) {
    RollAngle = cMin(RollAngle + cDegToRad(1.0), cDegToRad(180.0));
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
  window = glfwCreateWindow(w, h, "Force Dimension - Constraints Example", NULL, NULL);
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
  cout << "Force Dimension - OpenGL Constraints Example " << major << "." << minor << "." << release << "." << revision << endl;
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
  printf ("   'w' to toggle spherical workspace limit\n");
  printf ("   'c' to toggle conical rotational limit\n");
  printf ("   '1' to increase cone angle\n");
  printf ("   '2' to decrease cone angle\n");
  printf ("   '3' to increase roll angle\n");
  printf ("   '4' to decrease roll angle\n");
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
