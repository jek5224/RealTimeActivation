from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import numpy as np
import quaternion


def drawGround(height):
    
    glLineWidth(0.5); 
    glColor3f(0.0, 0.0, 0.0); 
    glDisable(GL_LIGHTING); 

    grid_vertices = np.ones((4 * 100, 3)); 
    grid_vertices[:, 1] = height; 

    grid_vertices[::4, 0] = 50; 
    grid_vertices[::4, 2] = np.arange(100) - 50; 

    grid_vertices[1::4, 0] = -50; 
    grid_vertices[1::4, 2] = np.arange(100) - 50; 

    grid_vertices[2::4, 0] = np.arange(100) - 50; 
    grid_vertices[2::4, 2] = 50; 

    grid_vertices[3::4, 0] = np.arange(100) - 50; 
    grid_vertices[3::4, 2] = -50; 

    glEnableClientState(GL_VERTEX_ARRAY); 
    glBindBuffer(GL_ARRAY_BUFFER, 0); 
    glVertexPointer(3, GL_FLOAT, 0, grid_vertices); 
    glDrawArrays(GL_LINES, 0, len(grid_vertices)); 
    glDisableClientState(GL_VERTEX_ARRAY); 
    glFlush(); 

    glEnable(GL_LIGHTING); 

def draw_axis(pos = np.array([0.0,0.0,0.0]), ori = np.quaternion(1.0, 0.0, 0.0, 0.0)):
    glPushMatrix()

    glTranslatef(pos[0], pos[1], pos[2])
    q = quaternion.as_rotation_vector(ori)
    glRotatef(np.rad2deg(np.linalg.norm(q)), q[0], q[1], q[2])

    glDisable(GL_LIGHTING)
    glColor3f(1.0, 0.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(1.0, 0.0, 0.0)
    glEnd()

    glColor3f(0.0, 1.0, 0.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 1.0, 0.0)
    glEnd()

    glColor3f(0.0, 0.0, 1.0)
    glBegin(GL_LINES)
    glVertex3f(0.0, 0.0, 0.0)
    glVertex3f(0.0, 0.0, 1.0)
    glEnd()
    glEnable(GL_LIGHTING)
    glPopMatrix()


def draw_sphere(radius):
    quadric = gluNewQuadric()
    gluQuadricDrawStyle(quadric, GLU_FILL)
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluQuadricTexture(quadric, GL_TRUE)
    gluSphere(quadric, radius, 50, 50)    


def draw_cube(size):
    glScaled(size[0], size[1], size[2])
    glutSolidCube(1.0)

