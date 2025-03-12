import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../flexiv_rdk/lib_py"))
import os
import glob
from my_device.camera import CameraD400
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import transforms3d
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import pybullet as p
import math
# from flexiv import FlexivRobot

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. eucledian norm, along axis.
    # >>> v0 = numpy.random.random(3)
    # >>> v1 = unit_vector(v0)
    # >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    # True
    # >>> v0 = numpy.random.rand(5, 4, 3)
    # >>> v1 = unit_vector(v0, axis=-1)
    # >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    # >>> numpy.allclose(v1, v2)
    # True
    # >>> v1 = unit_vector(v0, axis=1)
    # >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    # >>> numpy.allclose(v1, v2)
    # True
    # >>> v1 = numpy.empty((5, 4, 3), dtype=numpy.float64)
    # >>> unit_vector(v0, axis=1, out=v1)
    # >>> numpy.allclose(v1, v2)
    # True
    # >>> list(unit_vector([]))
    # []
    # >>> list(unit_vector([1.0]))
    # [1.0]
    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def rotation_matrix(angle, direction, point=None):
    """Return matrix to rotate about axis defined by point and direction.
    # >>> angle = (random.random() - 0.5) * (2*math.pi)
    # >>> direc = np.random.random(3) - 0.5
    # >>> point = np.random.random(3) - 0.5
    # >>> R0 = rotation_matrix(angle, direc, point)
    # >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    # >>> is_same_transform(R0, R1)
    # True
    # >>> R0 = rotation_matrix(angle, direc, point)
    # >>> R1 = rotation_matrix(-angle, -direc, point)
    # >>> is_same_transform(R0, R1)
    # True
    # >>> I = np.identity(4, np.float64)
    # >>> np.allclose(I, rotation_matrix(math.pi*2, direc))
    # True
    # >>> np.allclose(2., np.trace(rotation_matrix(math.pi/2,
    # ...                                                direc, point)))
    # True
    """
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vector(direction[:3])
    # rotation matrix around unit vector
    R = np.array(((cosa, 0.0,  0.0),
                     (0.0,  cosa, 0.0),
                     (0.0,  0.0,  cosa)), dtype=np.float64)
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array((( 0.0,         -direction[2],  direction[1]),
                      ( direction[2], 0.0,          -direction[0]),
                      (-direction[1], direction[0],  0.0)),
                     dtype=np.float64)
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    return M

def rotx(angle,Tc=False):
    Rx = rotation_matrix(angle, (1, 0, 0))
    if Tc:
        return Rx
    else:
        return Rx[0:3,0:3]

def roty(angle,Tc=False):
    Ry = rotation_matrix(angle, (0,1, 0))
    if Tc:
        return Ry
    else:
        return Ry[0:3,0:3]

def rotz(angle,Tc=False):
    Rz = rotation_matrix(angle, (0, 0, 1))
    if Tc:
        return Rz
    else:
        return Rz[0:3,0:3]



def rotation_from_matrix(matrix):
    """Return rotation angle and axis from rotation matrix.
    # >>> angle = (random.random() - 0.5) * (2*math.pi)
    # >>> direc = numpy.random.random(3) - 0.5
    # >>> point = numpy.random.random(3) - 0.5
    # >>> R0 = rotation_matrix(angle, direc, point)
    # >>> angle, direc, point = rotation_from_matrix(R0)
    # >>> R1 = rotation_matrix(angle, direc, point)
    # >>> is_same_transform(R0, R1)
    True
    """
    R = np.array(matrix, dtype=np.float64, copy=False)
    R33 = R[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, W = np.linalg.eig(R33.T)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    direction = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R33 corresponding to eigenvalue of 1
    l, Q = np.linalg.eig(R)
    i = np.where(abs(np.real(l) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]
    # rotation angle depending on direction
    cosa = (np.trace(R33) - 1.0) / 2.0
    if abs(direction[2]) > 1e-8:
        sina = (R[1, 0] + (cosa-1.0)*direction[0]*direction[1]) / direction[2]
    elif abs(direction[1]) > 1e-8:
        sina = (R[0, 2] + (cosa-1.0)*direction[0]*direction[2]) / direction[1]
    else:
        sina = (R[2, 1] + (cosa-1.0)*direction[1]*direction[2]) / direction[0]
    angle = math.atan2(sina, cosa)
    return angle, direction, point


def crossprod(a):
    '''
    [e2]x   = as defined on p554 =    [   0,-e3,e2;
                                       e3,0,-e1;
                                       -e2.e1.0    ]
    Input:     a       A Vector (3x1)
    Output:    ax      The matrix [a]x as defined above
    Syntax:    ax = crossprod(a)
    :param a:
    :return:
    '''
    a=np.asanyarray(a)
    ax=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
    return ax

def averageTransformation(H):
    '''
    :param H: [h] list, h is 4*4 pose.(np.array())
    :return: average pose h
    '''
    n=len(H)
    Ravg=[]
    for i in range(n):
        Ravg.append(H[i][0:3,0:3])
    R=averageRotation(Ravg)
    Tavg = 0
    for i in range(n):
        Tavg+=H[i][0:3,3]
    Havg=np.zeros((4,4))
    Havg[0:3,0:3]=R
    Havg[0:3,3]=Tavg/n
    Havg[3,3]=1
    return Havg

def averageRotation(R):
    n=len(R)
    tmp=np.zeros((3,3),dtype=R[0].dtype)
    for i in range(n):
        tmp=tmp+R[i]
    Rbarre=tmp/n
    RTR=np.dot(Rbarre.T,Rbarre) 
    [D,V]=np.linalg.eig(RTR) 
    D=np.diag(np.flipud(D))  
    V=np.fliplr(V)
    sqrtD=np.sqrt(np.linalg.inv(D)) 
    if np.linalg.det(Rbarre[0:3,0:3])<0:
        sqrtD[2,2]=-1*sqrtD[2,2]
    temp=np.dot(V,sqrtD)         
    temp=np.dot(temp,V.T)          
    Rvag=np.dot(Rbarre,temp)       
    return Rvag

def calibration_external(Hm2w,Hg2c):
    '''
    Hm2w,Hg2c , list of np.array((4,4))
    :param Hm2w: robot pose : robot end-effector frame in robot base
    :param Hg2c: robot outboard frame to robot inboard frame, e.g. "eye-to-hand", from camera to board. "eye-in-hand", from board to cam
    :return: return transform (from end-effector to robot inboard frame), and error
    '''
    assert type(Hm2w) is list,'Hm2w is not a list'
    assert type(Hg2c) is list,'Hg2c is not a list'
    # print Hm2w
    # print '.....'
    # print Hg2c
    num=len(Hm2w)
    Hg=[]
    Hc=[]
    A=np.array([])
    b = np.array([])
    for i in range(num-1):
        Hgij=np.dot(np.linalg.inv(Hm2w[i+1]),Hm2w[i])
        Hcij=np.dot(Hg2c[i+1],np.linalg.inv(Hg2c[i]))
        Hg.append(Hgij)
        Hc.append(Hcij)
        theta_gij,rngij,p1=rotation_from_matrix(Hgij)
        theta_cij,rncij,p2=rotation_from_matrix(Hcij)
        Pgij = 2 * math.sin(theta_gij / 2) * rngij
        Pcij = 2 * math.sin(theta_cij / 2) * rncij
        matrix_temp=crossprod(Pgij+Pcij)

        vector_temp=Pcij-Pgij
        vector_temp.shape=(3,1)
        if i==0:
            A=matrix_temp
            b=vector_temp
        else:
            A = np.concatenate((A, matrix_temp), axis=0)
            b = np.concatenate((b, vector_temp), axis=0)
    # Computing Rotation
    Pcg_prime=np.dot(np.linalg.pinv(A),b)
    err=np.dot(A,Pcg_prime)-b
    residus_TSAI_rotation=math.sqrt(np.sum(np.dot(np.transpose(err),err))/(num-1))
    Pcg = 2 * Pcg_prime / (math.sqrt(1 + np.linalg.norm(Pcg_prime)**2))
    Rcg = (1 - np.linalg.norm(Pcg) * np.linalg.norm(Pcg) / 2) * np.eye(3) +\
          0.5 * (np.dot(Pcg , np.transpose(Pcg))+math.sqrt(4-np.linalg.norm(Pcg)*np.linalg.norm(Pcg))*crossprod(Pcg))
    # Computing Translation
    A=[]
    b=[]
    # print 'Rcg',Rcg
    for i in range(num-1):
        matrix_temp=Hg[i][0:3,0:3]-np.eye(3)
        vector_temp=np.dot(Rcg,Hc[i][0:3,3])-Hg[i][0:3,3]
        vector_temp.shape=(3,1)
        if i==0:
            A=matrix_temp
            b=vector_temp
        else:
            A = np.concatenate((A, matrix_temp), axis=0)
            b = np.concatenate((b, vector_temp), axis=0)
    Tcg=np.dot(np.linalg.pinv(A),b)
    err=np.dot(A,Tcg)-b
    residus_TSAI_translation = math.sqrt(np.sum(np.dot(np.transpose(err),err))/(num-1))
    Hcam2marker_=np.zeros((4,4))
    Hcam2marker_[0:3,0:3]=Rcg
    Tcg.shape=(3,)
    Hcam2marker_[0:3,3]=Tcg
    Hcam2marker_[3,:]=np.array([0 ,0 ,0 ,1])
    err=[residus_TSAI_rotation,residus_TSAI_translation]
    return Hcam2marker_,err

def xyzrpy2tr(pose_list):
    T=np.identity(4,dtype=np.float64)
    T[0, 3] = pose_list[0] * 1000
    T[1, 3] = pose_list[1] * 1000
    T[2, 3] = pose_list[2] * 1000
    r=np.dot(rotz(pose_list[5]),roty(pose_list[4]))
    r=np.dot(r,rotx(pose_list[3]))
    T[0:3,0:3]=r
    return T
def fanuc_xyzrpy2tr(pose_list):
    T=np.identity(4,dtype=np.float64)
    T[0, 3] = pose_list[0]/1000
    T[1, 3] = pose_list[1]/1000
    T[2, 3] = pose_list[2]/1000
    pose_list[3]=pose_list[3]*math.pi/180.0
    pose_list[4]=pose_list[4]*math.pi/180.0
    pose_list[5]=pose_list[5]*math.pi/180.0
    r=np.dot(rotz(pose_list[5]),roty(pose_list[4]))
    r=np.dot(r,rotx(pose_list[3]))
    T[0:3,0:3]=r
    return T

def rodrigues_trans2tr(rvec, tvec):
    r,_=cv2.Rodrigues(rvec)
    tvec.shape=(3,)
    T=np.identity(4)
    T[0:3,3]=tvec
    T[0:3,0:3]=r
    return T

def XYZRodrigues_to_Hmatrix(pos):
    '''
    :param pos: [x,y,z,rodrigue]
    :return: np.array((4,4))
    '''
    xyz = pos[:3]
    rod = pos[3:6]
    rot_matrix,_ = cv2.Rodrigues(np.array(rod))
    Hmatrix = np.identity(4)
    Hmatrix[:3,:3] = rot_matrix
    Hmatrix[:3,3] = np.array(xyz).reshape(3)
    return Hmatrix

class Calib_extrinsic(object):
    def __init__(self, mtx, intial_pose,objpoints, imgpoints, Hm2w, hand_eye = 'eye_in_hand'):
        self.mtx = mtx                       # intrinsic, np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        self.intial_pose = intial_pose       # [x,y,z,rodrigues]+[x,y,z,rodrigues], end-effector -> camera + robot base -> calib
        self.objpoints = objpoints           # the 3d coordinates of cross points in calibration board frame
        self.imgpoints = imgpoints           # uv pixels of cross points in image
        self.Hm2w = Hm2w                     # robot pose
        self.hand_eye = hand_eye

    def obj_func(self,poss):
        '''
        optimize the object function using least-square method
        :param pos: camera pose from robot base, [x,y,z,rodrigues]
        :return:
        '''
        dist = (0.0, 0.0, 0.0, 0.0, 0.0)  # hack no distortion
        e2c = poss[0:6]  # x,y,z,rodrigues, end-effector -> camera
        b2g = poss[6:12]  # x,y,z,rodrigues, robot base -> calib
        Hc2m = XYZRodrigues_to_Hmatrix(e2c)
        Hgrid2worldAvg = XYZRodrigues_to_Hmatrix(b2g)
        mean_error_extrinsic = np.zeros(self.imgpoints[0].flatten().shape)
        mean_error_extrinsic_log = 0
        for i in range(len(self.objpoints)):
            # extrinsic param error
            temp = np.dot(self.Hm2w[i], Hc2m) # robot base -> camera
            temp = np.linalg.inv(temp)  # camera -> robot base
            cam_Matrix = np.dot(temp, Hgrid2worldAvg) # camera -> calib
            if self.hand_eye == 'eye_to_hand':
                cam_Matrix = np.linalg.inv(cam_Matrix) # calib -> camera
            elif self.hand_eye == 'eye_in_hand':
                cam_Matrix = cam_Matrix # camera -> calib
            cam_T = cam_Matrix[:3, 3]
            cam_T.shape = (1, 3)
            cam_R,_ = cv2.Rodrigues(cam_Matrix[:3, :3])
            imgpoints2_extrinsic, _ = cv2.projectPoints(self.objpoints[i], cam_R, cam_T*1000, self.mtx, dist)
            error_extrinsic_logs = cv2.norm(self.imgpoints[i], imgpoints2_extrinsic, cv2.NORM_L2) / len(imgpoints2_extrinsic)
            # pdb.set_trace()
            # error_extrinsic = abs(self.imgpoints[i] - imgpoints2_extrinsic).flatten(1)
            # mean_error_extrinsic += error_extrinsic
            mean_error_extrinsic_log += error_extrinsic_logs
        diff_log = mean_error_extrinsic_log/len(self.objpoints)
        #print(diff_log)
        # diff = mean_error_extrinsic
        return diff_log

    def run(self):
        final_pose = least_squares(self.obj_func,self.intial_pose)
        # pdb.set_trace()
        #print('--- original ---')
        #print(XYZRodrigues_to_Hmatrix(self.intial_pose[6:12]))
        #print('--- optimized ---')
        #print(XYZRodrigues_to_Hmatrix(final_pose.x[6:12]))
        return final_pose.x


def Hmatrix_to_XYZRodrigues(Hmatrix):
    rot_matrix = Hmatrix[:3,:3]
    xyz = Hmatrix[:3,3]
    rod,_ = cv2.Rodrigues(rot_matrix)
    return xyz.tolist()+rod.flatten().tolist()


class calibration():
    def __init__(self,mtx,pattern_size=(8,6),square_size=15,handeye='EIH'):
        '''

        :param image_list:  image array, num*720*1280*3
        :param pose_list: pose array, num*4*4
        :param pattern_size: calibration pattern size
        :param square_size: calibration pattern square size, 15mm
        :param handeye:
        '''
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.handeye = handeye
        self.pose_list = []
        self.joint_list = []
        self.init_calib()
        self.mtx = mtx

    def init_calib(self):
        self.objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        self.objp[:, :2] = self.square_size * np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        for i in range(self.pattern_size[0] * self.pattern_size[1]):
            x, y = self.objp[i, 0], self.objp[i, 1]
            self.objp[i, 0], self.objp[i, 1] = y, x
        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints = []  # 2d points in image plane.
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def detectFeature(self,color,show=True):
        img = color
        self.gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, corners = cv2.findChessboardCorners(img, self.pattern_size, None,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH)  # + cv2.CALIB_CB_NORMALIZE_IMAGE+ cv2.CALIB_CB_FAST_CHECK)
        if ret == True:
            self.objpoints.append(self.objp)
            # corners2 = corners
            if (cv2.__version__).split('.')[0] == '2':
                # pdb.set_trace()
                cv2.cornerSubPix(self.gray, corners, (5, 5), (-1, -1), self.criteria)
                corners2 = corners
            else:
                corners2 = cv2.cornerSubPix(self.gray, corners, (5, 5), (-1, -1), self.criteria)
            self.imgpoints.append(corners2)
            if show:
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                plt.title('img with feature point')
                for i in range(self.pattern_size[0] * self.pattern_size[1]):
                    ax.plot(corners2[i, 0, 0], corners2[i, 0, 1], color = "yellow", marker = 'o', markersize=12)
                plt.show()
        return ret

    def rodrigues_trans2tr(self,rvec, tvec):
        r, _ = cv2.Rodrigues(rvec)
        tvec.shape = (3,)
        T = np.identity(4)
        T[0:3, 3] = tvec
        T[0:3, 0:3] = r
        return T


    def cal(self,optimize=False):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], self.mtx, None)
        self.mtx = mtx # 这一行是加的，希望以新校准的intr为准。后面的代码其实混用了 self.mtx 和 mtx，不知道有没有精度问题
        print(mtx)
          #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], self.mtx, None)
        # Hm2w = []  # robot end-effector pose
        # import pdb
        # k = 0 
        # R,_=cv2.Rodrigues(rvecs[k])
        # for x,y in zip(self.objpoints[k],self.imgpoints[k]):
        #     z = self.mtx@(R@x+tvecs[k][:,0])
        #     z = z / z[2]
        #     print(z, y)
        #     pdb.set_trace()
        Hg2c = []
        pose_list = np.array(self.pose_list)
        for i in range(len(rvecs)):
            tt = self.rodrigues_trans2tr(rvecs[i], tvecs[i] / 1000.)
            Hg2c.append(tt)                    #camera2calib
        Hg2c = np.array(Hg2c)
        print("======================camera-to-goal=====================")
        #print(Hg2c[0])
        
        print("=======================world_to_end======================")
        print(pose_list[0])
        # for i in range(len(tvecs)):
        #     print(pose_list[i])

        
        print(pose_list.shape, Hg2c.shape)
        
        rot, pos = cv2.calibrateHandEye(pose_list[:, :3, :3], pose_list[:, :3, 3], Hg2c[:, :3, :3], Hg2c[:, :3, 3])
        camT = np.identity(4)
        camT[:3, :3] = rot
        camT[:3, 3] = pos[:,0]
        Hc2m = camT # gripper2camera
        print("===========================camT===========================")
        print(camT)
        if optimize:
            Hc2w = []
            Hw2c = []
            Hg2w = []
            Hw2g = []
            for i in range(len(rvecs)):
                temp = np.dot(pose_list[i], Hc2m)  # base2camera
                Hc2w.append(temp)
                Hw2c.append(np.linalg.inv(temp))   # camera2base
                temp = np.dot(temp, Hg2c[i])       # base2calib
                Hg2w.append(temp) 
                Hw2g.append(np.linalg.inv(temp))   # calib2base

            Hgrid2worldAvg = averageTransformation(Hg2w)   # base2calib
            mean_error = 0
            mean_error_extrinsic = 0
            # pdb.set_trace()
            for i in range(len(self.objpoints)):  
                imgpoints2, _ = cv2.projectPoints(self.objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error += error
                # pdb.set_trace()
                # extrisick param error
                temp = np.dot(pose_list[i], Hc2m)  # base2camera
                temp = np.linalg.inv(temp)         # camera2base
                cam_Matrix = np.dot(temp, Hgrid2worldAvg)  # camera2calib
                cam_Matrix = cam_Matrix          
                cam_T = cam_Matrix[:3, 3]
                cam_T.shape = (3, 1)              
                cam_R, _ = cv2.Rodrigues(cam_Matrix[:3, :3])  
                # pdb.set_trace()
                imgpoints2_extrinsic, _ = cv2.projectPoints(self.objpoints[i], cam_R, cam_T * 1000, mtx, dist)
                error_extrinsic = cv2.norm(self.imgpoints[i], imgpoints2_extrinsic, cv2.NORM_L2) / len(imgpoints2_extrinsic)
                mean_error_extrinsic += error_extrinsic  
                # pdb.set_trace()
            e2c = Hmatrix_to_XYZRodrigues(Hc2m) # e2camera
            b2g = Hmatrix_to_XYZRodrigues(Hgrid2worldAvg) # base2calib
            # pdb.set_trace()
            intial_pose = e2c + b2g
            cali = Calib_extrinsic(self.mtx, intial_pose, self.objpoints, self.imgpoints, pose_list, hand_eye='eye_in_hand')
            final_pose = cali.run()
            for i in range(3):
                intial_pose = final_pose
                cali = Calib_extrinsic(self.mtx, intial_pose, self.objpoints, self.imgpoints, pose_list, hand_eye='eye_in_hand')
                final_pose = cali.run()
            Hc2m = XYZRodrigues_to_Hmatrix(final_pose[:6])
            camT = Hc2m
   
        # q_gt = np.array([0.07717, 0, 0.0365, 0.7071068, 0, 0, 0.7071067])
        # camT_gt = np.eye(4)
        # camT_gt[:3,:3] = transforms3d.quaternions.quat2mat(q_gt[3:]).T
        # camT_gt[:3,3] = q_gt[:3]
        # print("========== camgt ===================")
        # print(camT_gt)
  
        for i in range(len(pose_list)):
            ppp = pose_list[i] @ camT @ Hg2c[i]
            # print(ppp)
            p.addUserDebugLine(ppp[:3,3], ppp[:3,3] + ppp[0,:3]*0.1, (1,0,0), 3.0)
            p.addUserDebugLine(ppp[:3,3], ppp[:3,3] + ppp[1,:3]*0.1, (0,1,0), 3.0)
            p.addUserDebugLine(ppp[:3,3], ppp[:3,3] + ppp[2,:3]*0.1, (0,0,1), 3.0)

        return camT # Optimized end-effector2camera transformation




if __name__ == "__main__":
    cam = CameraD400(serial="104422070044")
    physicsClient = p.connect(p.GUI)
    rob = p.loadURDF("../flexiv_rdk/resources/flexiv_rizon4_kinematics.urdf")
    # flexiv = FlexivRobot(robot_ip_address="192.168.2.100",
    #                      pc_ip_address="192.168.2.35")
    # flexiv.switch_mode('cartesian')
    # a = input("ready")

    intr = cam.getIntrinsics()
    # intr = np.array([[442.13515065, 0., 353.26175986], # calibrate
    #                  [0., 440.36922242, 205.60118088],
    #                  [0., 0., 1.]])
    print(intr)
    calib = calibration(intr, (11, 8), 10)
    dirName = 'cali_hand'
    flangeposList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/*t.txt")
    flangeposList.sort()
    jointList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/*j.txt")
    jointList.sort()
    imgList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/*c.png")
    imgList.sort()
    print(len(flangeposList), len(imgList))
    # pose_list = []
    # flangeposList = flangeposList[7:]
    # imgList = imgList[7:]
    for i, filename in enumerate(flangeposList):
        if len(calib.objpoints) >100:
            break
        assert(imgList[i][:-5] == filename[:-5])
        ee_pose_tq = np.loadtxt(filename).tolist()
        joint = np.loadtxt(jointList[i]).tolist()

        # ee_pose_tq = flexiv.tcp_pose
        ee_pose_t = ee_pose_tq[:3]
        ee_pose_q = ee_pose_tq[3:]
        ee_pose_r = transforms3d.quaternions.quat2mat(ee_pose_q)
        
          # ee_pose_r: [w, x, y, z]
        #ee_pose_r = R.from_quat([ee_pose_q[3], ee_pose_q[0], ee_pose_q[1], ee_pose_q[2]]).as_matrix()
  
        current_pose = np.eye(4)
        current_pose[:3,:3] = ee_pose_r
        current_pose[:3,3] = ee_pose_t
        # print(current_pose)
        # time.sleep(1)
        # color, depth = cam.get_data() # output rgb and depth image
        
        color = cv2.imread(imgList[i])
  
        # while True:
        #     cv2.imshow('color', color)
        #     cv2.waitKey(1)
        #     color, depth = cam.get_data()  # output rgb and depth image

        flag = calib.detectFeature(color, show=False)
        if flag:
            calib.pose_list.append(current_pose)
            calib.joint_list.append(joint)
        else:
            print(" [INFO] No chessboard detected in image ", imgList[i])
        # os.makedirs(calib_image_root, exist_ok=True)
        # os.makedirs(osp.join(calib_image_root, 'color'), exist_ok=True)
        # os.makedirs(osp.join(calib_image_root, 'depth'), exist_ok=True)
        # cv2.imwrite(osp.join(calib_image_root, 'depth', 'franka_%0.3d.png' % (i + 1)), depth)
        # cv2.imwrite(osp.join(calib_image_root, 'color', 'franka_%0.3d.png' % (i + 1)), color)
        # pose_list.append(current_pose)
    print(len(calib.objpoints))
    camT = calib.cal(optimize=True)
    print("="*10)
    print(camT)

    # while True:
    #     a = 0
    np.save(os.path.join(f'./workspace/flexiv/calib/out/{dirName}/','intrinsic.npy'), calib.mtx)
    np.save(os.path.join(f'./workspace/flexiv/calib/out/{dirName}/','camT.npy'), camT)
    # np.save('config/franka_campose.npy',camT)
    print("saved")
    while True:
        p.stepSimulation()
