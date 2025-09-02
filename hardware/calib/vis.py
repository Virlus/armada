import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../flexiv_rdk/lib_py"))
import numpy as np
import glob 
import transforms3d
import cv2
from PIL import Image
from calib.utils import getXYZRGB
import open3d as o3d

WORKSPACE_MIN = np.array([-0.5, -0.5, 0])
WORKSPACE_MAX = np.array([0.5, 0.5, 1.0])

# np read from file
intrinsic = np.load('./workspace/flexiv/calib/out/cali_hand/intrinsic.npy')
camT = np.load('./workspace/flexiv/calib/out/cali_hand/camT.npy')
camsideT = np.load('./workspace/flexiv/calib/out/cali_side/camT.npy')
intrinsicside = np.load('./workspace/flexiv/calib/out/cali_side/intrinsic.npy')

# intrinsic = np.array(
#     [[605.66461182,   0.        , 320.4206543 ],
#        [  0.        , 605.10345459, 234.22007751],
#        [  0.        ,   0.        ,   1.        ]]
# )

# intrinsic = np.array(
# [[902.9407168407078, 0, 637.0184314889033],
#  [0, 903.1485236556671, 346.1738529498695],
#  [0, 0, 1]]
# )
# camT = np.array(
# [[0.028796754768446946, 0.9948221235791507, -0.09746634984585564, -0.08124059504734372],
#  [-0.9965503781120361, 0.02098005259260349, -0.08029434150654285, 0.002160432146839692],
#  [-0.07783373818315337, 0.09944234425374529, 0.9919945208365597, -0.17110154793933882],
#  [0, 0, 0, 1]]
# )

# camsideT = np.array(
#     [[-0.020743092022448284, 0.954259172984447, -0.29826021341839626, 0.8854264018224021],
#  [0.9972788591187729, -0.001360046312434804, -0.07370907290684475, 0.13811837605009536],
#  [-0.0707432066569417, -0.298977559440616, -0.9516342877717394, 0.7482671728960535],
#  [0, 0, 0, 1]]
# )
# intrinsicside = np.array(
# [[896.1683091253756, 0, 656.5452548066794],
# [0, 896.2158076614431, 361.2637154479055],
# [0, 0, 1]]
# )

Hand = True
# Hand = False

g_list= []
if Hand:
    dirName = 'cali_hand'
    flangeposList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/*t.txt")
    flangeposList.sort()
    imgList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/*c.png")
    imgList.sort()
    depthList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/*d.png")
    depthList.sort()
else:
    dirName = 'cali_side'
    flangeposList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/*.txt")
    flangeposList.sort()
    imgList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/hand/*c.png")
    imgList.sort()
    depthList = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/hand/*d.png")
    depthList.sort()
    imgList2 = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/side/*c.png")
    imgList2.sort()
    depthList2 = glob.glob(f"./workspace/flexiv/calib/out/{dirName}/side/*d.png")
    depthList2.sort()

for i, filename in enumerate(flangeposList):
    if len(flangeposList) > 30 and np.random.rand()< 0.9:
        continue
    ee_pose_tq = np.loadtxt(filename).tolist()
    print(ee_pose_tq)
    ee_pose_t = ee_pose_tq[:3]
    ee_pose_q = ee_pose_tq[3:]
    ee_pose_r = transforms3d.quaternions.quat2mat(ee_pose_q)


    current_pose = np.eye(4)
    current_pose[:3,:3] = ee_pose_r
    current_pose[:3,3] = ee_pose_t
    g2w = np.linalg.inv(current_pose) # gripper2world

    color = cv2.imread(imgList[i])
    depth = np.array(Image.open(depthList[i]), dtype = np.float32)
    
    xyzrgb = getXYZRGB(color, depth, current_pose, camT, intrinsic)
    points = xyzrgb[:,:3]
    colors = xyzrgb[:,3:]


    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
    g_list.extend([pcd, o])

    if not Hand:
        color2 = cv2.imread(imgList2[i])
        depth2 = np.array(Image.open(depthList2[i]), dtype = np.float32)
        xyzrgb2 = getXYZRGB(color2, depth2, np.eye(4), camsideT, intrinsicside, )
        points2 = xyzrgb2[:,:3]
        colors2 = xyzrgb2[:,3:]
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.colors = o3d.utility.Vector3dVector(colors2)
        g_list.extend([pcd2])

    # cam = o3d.geometry.TriangleMesh.create_sphere(0.01).translate(camO)
    # c = o3d.geometry.TriangleMesh.create_coordinate_frame(0.05).transform(ppp)
    # g_list.extend([c])

    # Add desktop frame
    points = [
        [0.3, -0.4, 0],
        [0.9, -0.4, 0],
        [0.3, 0.4, 0],
        [0.9, 0.4, 0],
    ]
    lines = [
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 3],
    ]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    g_list.extend([line_set])

    if not Hand:
        o3d.visualization.draw_geometries(g_list)
        g_list = []

if Hand:
    o3d.visualization.draw_geometries(g_list)
    g_list = []