import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../flexiv_rdk/lib_py"))
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
        self.intial_pose = intial_pose       # [x,y,z,rodrigues]+[x,y,z,rodrigues], end-effector -> calib + robot base -> camera
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
        e2g = poss[0:6]  # x,y,z,rodrigues, end-effector -> calib
        b2c = poss[6:12]  # x,y,z,rodrigues, robot base -> camera
        Hc2m = XYZRodrigues_to_Hmatrix(e2g)
        Hgrid2worldAvg = XYZRodrigues_to_Hmatrix(b2c)
        mean_error_extrinsic = np.zeros(self.imgpoints[0].flatten().shape)
        mean_error_extrinsic_log = 0
        for i in range(len(self.objpoints)):
            # extrinsic param error
            temp = np.dot(self.Hm2w[i], Hc2m)
            temp = np.linalg.inv(temp)
            cam_Matrix = np.dot(temp, Hgrid2worldAvg)
            if self.hand_eye == 'eye_to_hand':
                cam_Matrix = np.linalg.inv(cam_Matrix)
            elif self.hand_eye == 'eye_in_hand':
                cam_Matrix = cam_Matrix
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

    def get_Hg2c(self):
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], self.mtx, None)
        Hg2c = []
        for i in range(len(rvecs)):
            tt = self.rodrigues_trans2tr(rvecs[i], tvecs[i] / 1000.)
            Hg2c.append(tt)
        Hg2c = np.array(Hg2c)
        return Hg2c

    

    def cal(self, calib_wrist):
        def transformation_error_func(params, obj2base, obj2cam):
            tx, ty, tz, qx, qy, qz, qw = params
            estimated_transform = transforms3d.quaternions.quat2mat([qw, qx, qy, qz])
            estimated_transform = np.vstack((estimated_transform, [0, 0, 0]))
            estimated_transform = np.hstack((estimated_transform, [[tx], [ty], [tz], [1]]))
            
            error = []
            
            for i in range(len(obj2base)):
                # Compute the error between the observed transformation and the one computed by the estimated parameters
                observed_transform = obj2base[i] @ np.linalg.inv(obj2cam[i]) # base2sidecamera
                diff_transform = observed_transform @ np.linalg.inv(estimated_transform)
                
                # Calculate translational error
                translational_error = np.linalg.norm(diff_transform[:3, 3])
                
                # Calculate rotational error (as an angle)
                _, angle = transforms3d.axangles.mat2axangle(diff_transform[:3, :3])
                
                # Accumulate errors
                error.append(translational_error)
                error.append(angle)
            
            return error
        
        # camera_pose = np.array([[ 0.00886583,  0.99944172,  0.03221257, -0.07624523],
        #                         [-0.99975617,  0.00951093, -0.01992849,  0.06389011],
        #                         [-0.02022373, -0.03202803,  0.99928235,  0.00224033],
        #                         [ 0.,  0.,  0.,  1.]])
        # camera_pose = np.array(
        # [[ 0.03089147, 0.99886808, -0.03617001, -0.05927289],
        # [-0.99915533, 0.02987879, -0.02821134,  0.02904617],
        # [-0.02709869, 0.03701095,  0.99894737, -0.20291156],
        # [ 0.        , 0.        ,  0.        ,  1.        ],]
        # )

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.gray.shape[::-1], self.mtx, None)
        print('intrinsic', mtx)
        self.mtx = mtx
        
        camera_pose = np.load('./workspace/flexiv/calib/out/cali_hand/camT.npy')
    
        pose_list = np.array(self.pose_list)
        Hg2c = calib_wrist.get_Hg2c() # hand: camera2calib
        # obj2base = pose_list[0] @ camera_pose @ Hg2c
        obj2base = pose_list @ camera_pose @ Hg2c # base2calib
        # import pdb;pdb.set_trace()
        obj2cam = self.get_Hg2c() # side: camera2calib
        print(obj2base[0])

        camT = []
        for i in range(len(pose_list)):
            ppp = (obj2base[i]) @ np.linalg.inv(obj2cam[i]) # base2camera
            camT.append(ppp)
            p.addUserDebugLine(ppp[:3,3], ppp[:3,3] + ppp[0,:3]*0.1, (1,0,0), 3.0)
            p.addUserDebugLine(ppp[:3,3], ppp[:3,3] + ppp[1,:3]*0.1, (0,1,0), 3.0)
            p.addUserDebugLine(ppp[:3,3], ppp[:3,3] + ppp[2,:3]*0.1, (0,0,1), 3.0)
        
        initial_guess = [0, 0, 0, 0, 0, 0, 1] # Assume no initial rotation and translation

        print(obj2base.shape)
        res = least_squares(transformation_error_func, initial_guess, args=(obj2base, obj2cam))

        optimized_quaternion = [res.x[6], res.x[3], res.x[4], res.x[5]] # [qw, qx, qy, qz]
        optimized_rotation = transforms3d.quaternions.quat2mat(optimized_quaternion)
        optimized_translation = res.x[:3]

        optimized_cam2_to_base_transform = np.vstack((optimized_rotation, [0, 0, 0]))
        optimized_cam2_to_base_transform = np.hstack((optimized_cam2_to_base_transform, [[optimized_translation[0]], [optimized_translation[1]], [optimized_translation[2]], [1]]))

        camT = optimized_cam2_to_base_transform # base2sidecam
        # camT = np.mean(camT, axis=0)
        return camT


if __name__ == "__main__":
    # cam = CameraD400('135122075425')
    # cam = CameraD400('242322072982')
 
    physicsClient = p.connect(p.GUI)
    rob = p.loadURDF("../flexiv_rdk/resources/flexiv_rizon4_kinematics.urdf")
    # intr = cam.getIntrinsics()
    intr = np.load('./workspace/flexiv/calib/out/cali_hand/intrinsic.npy')

    pattern_size = (11, 8)
    calib = calibration(intr, pattern_size, 10)
    calib_wrist = calibration(intr, pattern_size, 10)
    dirName = 'cali_side'
    flangeposList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/*.txt")
    flangeposList.sort()
    imgwristList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/hand/*c.png")
    imgwristList.sort()
    
    imgList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/side/*c.png")
    imgList.sort()
    print(len(flangeposList),len(imgwristList), len(imgList))
    for i, _ in enumerate(flangeposList):
        ee_pose_tq = np.loadtxt(flangeposList[i]).tolist()
        
        ee_pose_t = ee_pose_tq[:3]
        ee_pose_q = ee_pose_tq[3:]
        ee_pose_r = transforms3d.quaternions.quat2mat(ee_pose_q)
        current_pose = np.eye(4)
        current_pose[:3,:3] = ee_pose_r
        current_pose[:3,3] = ee_pose_t
        
        color = cv2.imread(imgList[i])
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        color_ = cv2.imread(imgwristList[i])
        color_ = cv2.cvtColor(color_, cv2.COLOR_RGB2BGR)
        # cv2.imshow('a',color_)
        # cv2.waitKey(10)
        flag, _ = cv2.findChessboardCorners(color, pattern_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
        flag_, _ = cv2.findChessboardCorners(color_, pattern_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
  
        if flag and flag_:
            flag = calib.detectFeature(color, show=False)
            flag_ = calib_wrist.detectFeature(color_, show=False)
            calib.pose_list.append(current_pose)

        else:
            print(flag, flag_)
            print(" [INFO] No chessboard detected in image ", imgList[i])
    
    print(len(calib.objpoints))
    camT = calib.cal(calib_wrist) # base2sidecam
    print("="*10)
    print(camT)
    np.save(os.path.join(f'./workspace/flexiv/calib/out/{dirName}/','intrinsic.npy'), calib.mtx)
    np.save(os.path.join(f'./workspace/flexiv/calib/out/{dirName}/','camT.npy'), camT)
    print("saved")
    while True:
        p.stepSimulation()