///////////////////////////////////////////////////////////////////////////////
//
//  Copyright (C) 2001-2020 Force Dimension
//  All Rights Reserved.
//
///////////////////////////////////////////////////////////////////////////////



#ifndef CMacrosGLH
#define CMacrosGLH



#if defined(WIN32) || defined(WIN64)
#include "windows.h"
#endif
#include "Eigen/Eigen"

// GLU library
#ifdef MACOSX
#include "OpenGL/glu.h"
#else
#include "GL/glu.h"
#endif



//  Matrix class used to express position or translation.
//  On the OpenGL side 4x4 matrices are required to perform all
//  geometric transformations. cMatrixGL provides a structure
//  which encapsulates all the necessary functionalities to
//  generate 4x4 OpenGL transformation matrices by passing 3D
//  position vectors and rotation matrices. cMatrixGL also provides
//  OpenGL calls to push, multiply and pop matrices off the OpenGL stack.
//  OpenGL Matrices are COLUMN major.

struct cMatrixGL
{
private:

    double  m[4][4];

public:

    // returns a pointer to the matrix array in memory
    const double* pMatrix () const { return m[0]; }

    // creates OpenGL translation matrix from a position vector passed as parameter
    void set (const Eigen::Vector3d& a_pos) {
        m[0][0] = 1.0;        m[0][1] = 0.0;         m[0][2] = 0.0;         m[0][3] = 0.0;
        m[1][0] = 0.0;        m[1][1] = 1.0;         m[1][2] = 0.0;         m[1][3] = 0.0;
        m[2][0] = 0.0;        m[2][1] = 0.0;         m[2][2] = 1.0;         m[2][3] = 0.0;
        m[3][0] = a_pos (0);  m[3][1] = a_pos (1);   m[3][2] = a_pos (2);   m[3][3] = 1.0;
    }

    // creates OpenGL translation matrix from vector giving translation
    void set (const Eigen::Vector3d& a_pos, const double a_rot[3][3]) {
        m[0][0] = a_rot[0][0];   m[0][1] = a_rot[1][0];   m[0][2] = a_rot[2][0];   m[0][3] = 0.0;
        m[1][0] = a_rot[0][1];   m[1][1] = a_rot[1][1];   m[1][2] = a_rot[2][1];   m[1][3] = 0.0;
        m[2][0] = a_rot[0][2];   m[2][1] = a_rot[1][2];   m[2][2] = a_rot[2][2];   m[2][3] = 0.0;
        m[3][0] = a_pos (0);     m[3][1] = a_pos (1);     m[3][2] = a_pos (2);     m[3][3] = 1.0;
    }

    // creates OpenGL translation matrix from vector giving translation
    void set (const Eigen::Vector3d& a_pos, const Eigen::Matrix3d& a_rot) {
        m[0][0] = a_rot (0, 0);  m[0][1] = a_rot (1, 0);  m[0][2] = a_rot (2, 0);  m[0][3] = 0.0;
        m[1][0] = a_rot (0, 1);  m[1][1] = a_rot (1, 1);  m[1][2] = a_rot (2, 1);  m[1][3] = 0.0;
        m[2][0] = a_rot (0, 2);  m[2][1] = a_rot (1, 2);  m[2][2] = a_rot (2, 2);  m[2][3] = 0.0;
        m[3][0] = a_pos (0);     m[3][1] = a_pos (1);     m[3][2] = a_pos (2);     m[3][3] = 1.0;
    }

    // push current OpenGL on the stack and multiply with this cMatrixGL matrix
    void glMatrixPushMultiply () {
        glPushMatrix ();
        glMultMatrixd ((const double *)pMatrix ());
    }

    // pop current OpenGL matrix off the stack
    void glMatrixPop () {
        glPopMatrix ();
    }
};



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



#endif
