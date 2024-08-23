import argparse
import numpy as np
import cv2
import math
import os
from imageDepthEstimation import imageDepthEstimation
import time
import trimesh
import coacd

num = 351

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--left', dest='left',
                        help='left image path',
                        default='depth.png', type=str)
    parser.add_argument('--right', dest='right',
                    help='right image path',
                    default='depth.png', type=str)
    
    # parser.add_argument('--depthInvert', dest='depthInvert',
    #                     help='Invert depth map',
    #                     default=False, action='store_true')
    # parser.add_argument('--texturePath', dest='texturePath',
    #                     help='corresponding image path',
    #                     default='', type=str)
    # parser.add_argument('--objPath', dest='objPath',
    #                     help='output path of .obj file',
    #                     default='model.obj', type=str)
    # parser.add_argument('--mtlPath', dest='mtlPath',
    #                     help='output path of .mtl file',
    #                     default='model.mtl', type=str)
    # parser.add_argument('--matName', dest='matName',
    #                     help='name of material to create',
    #                     default='colored', type=str)

    args = parser.parse_args()
    return args

def create_mtl(mtlPath, matName, texturePath):
    if max(mtlPath.find('\\'), mtlPath.find('/')) > -1:
        os.makedirs(os.path.dirname(mtlPath), exist_ok=True)
    with open(mtlPath, "w") as f:
        f.write("newmtl " + matName + "\n"      )
        f.write("Ns 10.0000\n"                  )
        f.write("d 1.0000\n"                    )
        f.write("Tr 0.0000\n"                   )
        f.write("illum 2\n"                     )
        f.write("Ka 1.000 1.000 1.000\n"        )
        f.write("Kd 1.000 1.000 1.000\n"        )
        f.write("Ks 0.000 0.000 0.000\n"        )
        f.write("map_Ka " + texturePath + "\n"  )
        f.write("map_Kd " + texturePath + "\n"  )

def vete(v, vt):
    return str(v)+"/"+str(vt)

def create_obj(depthPath, objPath, depthInvert):
    
    img = cv2.imread(depthPath, -1)
    print(img.dtype)
    img = img.astype(np.float32)
    print(img.dtype)

    if len(img.shape) > 2 and img.shape[2] > 1:
       print('Expecting a 1D map, but depth map at path %s has shape %r'% (depthPath, img.shape))
       return

    if depthInvert == True:
        img = 1.0 - img

    h = img.shape[0]
    w = img.shape[1]

    FOV = math.radians(23)
    D = (w/2)/math.tan(FOV/2)

    if max(objPath.find('\\'), objPath.find('/')) > -1:
        os.makedirs(os.path.dirname(objPath), exist_ok=True)
    
    with open(objPath,"w") as f:    
        # if useMaterial:
        #     f.write("mtllib " + mtlPath + "\n")
        #     f.write("usemtl " + matName + "\n")

        ids = np.zeros((img.shape[1], img.shape[0]), int)
        vid = 1

        for u in range(0, w):
            for v in range(h-1, -1, -1):

                d = img[v, u]

                ids[u,v] = vid
                if d == 0.0:
                    ids[u,v] = 0
                vid += 1

                x = u - w/2
                y = v - h/2
                z = -D

                norm = 1 / math.sqrt(x*x + y*y + z*z)

                t = d/(z*norm)

                x = -t*x*norm
                y = t*y*norm
                z = -t*z*norm        

                f.write("v " + str(x) + " " + str(y) + " " + str(z) + "\n")

        for u in range(0, img.shape[1]):
            for v in range(0, img.shape[0]):
                f.write("vt " + str(u/img.shape[1]) + " " + str(v/img.shape[0]) + "\n")

        for u in range(0, img.shape[1]-1):
            for v in range(0, img.shape[0]-1):

                v1 = ids[u,v]; v2 = ids[u+1,v]; v3 = ids[u,v+1]; v4 = ids[u+1,v+1];

                if v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0:
                    continue

                f.write("f " + vete(v1,v1) + " " + vete(v2,v2) + " " + vete(v3,v3) + "\n")
                f.write("f " + vete(v3,v3) + " " + vete(v2,v2) + " " + vete(v4,v4) + "\n")

def object_simplification(objPath, num):

    mesh = trimesh.load(objPath, force="mesh")
    mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    result = coacd.run_coacd(
        mesh,
        threshold=0.04 # depend on user's choice
    )
    mesh_parts = []
    for vs, fs in result:
        mesh_parts.append(trimesh.Trimesh(vs, fs))
    scene = trimesh.Scene()
    np.random.seed(0)
    for p in mesh_parts:
        # p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        scene.add_geometry(p)
    
    output_path = '/home/dyros/tocabi_ws/src/Depth_Map_Visualizer/HITNET-Stereo-Depth-estimation/Object_coacd/flythings_obj_coacd' + str(num)+'.obj'
    scene.export(output_path)
    # print(os.path.abspath(output_path))
    print("CoACD object file created!")

    # return output_path

if __name__ == '__main__':
    print("STARTED")
    # args = parse_args()
    # useMat = args.texturePath != ''
    # if useMat:
    #     create_mtl(args.mtlPath, args.matName, args.texturePath)
    start_time = time.time()
    left_path = "/home/dyros/tocabi_ws/src/tocabi/data/stereo_seg/left/seg_left_real_e50_conf75_"+str(num)+".png"
    right_path = "/home/dyros/tocabi_ws/src/tocabi/data/stereo_seg/right/seg_right_real_e50_conf75_"+str(num)+".png"
    depthPath, max_dist = imageDepthEstimation(left_path, right_path, num)

    if depthPath.find("middlebury") != -1:
        objPath = "Object/middlebury_obj_raw"+str(num)+".obj"
    elif depthPath.find("fly") != -1:
        objPath = "Object/flythings_obj_seg_"+str(num)+"_"+str(max_dist)+".obj"
    elif depthPath.find("eth") != -1:
        objPath = "Object/eth3d_obj_raw"+str(num)+".obj"
    
    create_obj(depthPath, objPath, depthInvert=False)
    # object_simplification(objPath, num)

    print("FINISHED")
    end_time = time.time()
    print('total time consumed : ',end_time-start_time)
