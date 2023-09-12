import openmesh as om
import numpy as np
import math
from shapely.geometry import Polygon
import os

young = 5.4
possion = 0.33
mu = young/(2*(1+possion))
lamda = young*possion/((1+possion)*(1-2*possion))


"""
get Green Strain Energy function 

Input : raw_position,new_position,face,mu,lamda(two Lame parameters) 

"""

def Green_Strain_Energy(position_1,position_2,face_id_1,face_id_2):
    v1_index, v2_index, v3_index = face_id_1[0], face_id_1[1], face_id_1[2]
    v1_index_2, v2_index_2, v3_index_2 = face_id_2[0], face_id_2[1], face_id_2[2]
    #
    # W = trimesh_3Dto2D(mesh,position_1,face_id,face)

    W = np.zeros((2, 2))
    W_ = np.zeros((3, 2))
    W[:, 0] = (position_1[v2_index,0:2] - position_1[v1_index,0:2]).reshape((2,))
    W[:, 1] = (position_1[v3_index,0:2] - position_1[v1_index,0:2]).reshape((2,))
    W_[:, 0] = (position_2[v2_index_2] - position_2[v1_index_2]).reshape((3,))
    W_[:, 1] = (position_2[v3_index_2] - position_2[v1_index_2]).reshape((3,))

    Q, R = np.linalg.qr(W)
    tmp = np.dot(W_, np.linalg.inv(R))
    J_j = np.dot(tmp, Q.T)

    green_strain = 1/2*(np.dot(J_j.T,J_j)-np.eye(np.dot(J_j.T,J_j).shape[0]))
    energy = mu*(np.linalg.norm(green_strain,ord='fro')**2)+lamda*(green_strain.trace()**2)
    return energy


def get_weight(mesh,mesh2,neibor_face):

    max_energy = 50

    faces = mesh.faces()
    faces_id = mesh.fv_indices()
    list1 = []
    for v in mesh.vertices():
        list1.append(mesh.point(v))
    position_1 = np.array(list1)

    faces_id2 = mesh2.fv_indices()
    list2 = []
    for v in mesh2.vertices():
        list2.append(mesh2.point(v))
    position_2 = np.array(list2)


    energy_face = np.zeros(faces_id2.shape[0])


    outlier_set = []



    for i in range(faces_id.shape[0]):
        face_id_1 = faces_id[i,:]
        face_id_2 = faces_id2[i,:]
        energy = Green_Strain_Energy(position_1,position_2,face_id_1,face_id_2)
        if energy > max_energy:
            outlier_set.append(i)
        v1_index, v2_index, v3_index = face_id_1[0], face_id_1[1], face_id_1[2]
        a = np.linalg.norm((position_1[v2_index] - position_1[v1_index]).reshape((3,)),2)
        b = np.linalg.norm((position_1[v3_index] - position_1[v1_index]).reshape((3,)),2)
        c = np.linalg.norm((position_1[v3_index] - position_1[v2_index]).reshape((3,)),2)
        p = 1/2*(a+b+c)
        area = math.sqrt(p*(p-a)*(p-b)*(p-c))
        energy_face[i] = energy
        # weight_face[i] = energy*area

    for m in range(1):
        for index in outlier_set:
            neibor_energy = []
            neibor_set = []
            neibor_set.append(index)
            for i in range(2):
                for facce in neibor_set.copy():
                    # print(neibor_set)
                    for tmp_face in neibor_face[facce]:
                        neibor_set.append(tmp_face)
            for i in neibor_set:
                neibor_energy.append(energy_face[i])
            # print(np.mean(neibor_energy))
            energy_face[index] = np.min(neibor_energy)
    # plt.scatter(x=range(energy_face.shape[0]),y=energy_face)
    # plt.show()
    return energy_face
""" Test """





def calculate_weight_sequence(model_path,action_sequence_path,threeD_path):
    mesh = om.read_trimesh(model_path)
    list1 = []
    for v in mesh.vertices():
        list1.append(mesh.point(v))
    position = np.array(list1)

    mesh_3D = om.read_trimesh(threeD_path)
    edges_3D = mesh_3D.ev_indices()
    faces_3D = mesh_3D.fv_indices()
    faces = mesh_3D.faces()
    list1 = []
    for v in mesh_3D.vertices():
        list1.append(mesh_3D.point(v))
    position_3D = np.array(list1)

    neibor_face = {}
    for face in faces:
        id_set = []
        for tmp in mesh.fv(face):
            id_set.append(tmp.idx())
        neibor_face[face.idx()] = id_set

    energy_face = np.zeros(faces_3D.shape[0])
    path1 = action_sequence_path
    filelist = os.listdir(path1)
    frame = 0
    max = []
    min = []
    median = []
    for filename in filelist:
       filepath = os.path.join(path1, filename)
       if '.obj' in filepath:
            mesh2 = om.read_trimesh(filepath)
            t4 = get_weight(mesh,mesh2,neibor_face)
            energy_face += t4
            max_frame = np.max(t4)
            min_frame = np.min(t4)
            median_frame = np.median(t4)
            print(frame,max_frame,min_frame,median_frame)
            max.append(max_frame)
            min.append(min_frame)
            median.append(median_frame)
            frame += 1
    energy_face /= frame
    print(max)
    print(min)

    return energy_face

def get_edge_idx(mesh,path):
    edges = mesh.ev_indices()
    edges = edges[edges[:, 0].argsort()]
    path = np.array(path)
    edge_idx = np.zeros(len(path),dtype=int)
    for j in range(path.shape[0]):
        for i in range(edges.shape[0]):
            if path[j,0] == edges[i,0]:
                if path[j,1] == edges[i,1]:
                    edge_idx[j] = i
            elif path[j,0] == edges[i,1] :
                if path[j,1] == edges[i,0]:
                    edge_idx[j] = i
    return edge_idx

def get_face_area(mesh):
    faces = mesh.fv_indices()
    list1 = []
    for v in mesh.vertices():
        list1.append(mesh.point(v))
    position = np.array(list1)
    area = np.zeros(faces.shape[0])
    for index in range(faces.shape[0]):
        a = np.linalg.norm(position[faces[index,0]]-position[faces[index,1]])
        b = np.linalg.norm(position[faces[index, 0]] - position[faces[index, 2]])
        c = np.linalg.norm(position[faces[index, 1]] - position[faces[index, 2]])
        p = (a+b+c)/2
        area[index] = math.sqrt(p*(p-a)*(p-b)*(p-c))
    return area


def weight_facetoedge_2D(mesh,energy_face,length = 5/2):

    edges = mesh.ev_indices()
    edges = edges[edges[:, 0].argsort()]
    weight = np.zeros(edges.shape[0])
    area = np.zeros(edges.shape[0])
    list1 = []
    for v in mesh.vertices():
        list1.append(mesh.point(v))
    position = np.array(list1)
    face = mesh.faces()
    face_idx = mesh.fv_indices()
    visited = np.zeros(edges.shape[0])
    for f in face:
        print(f.idx())
        v_set = []
        for v in mesh.fv(f):
            v_set.append(v)
        edge_set = [[v_set[0],v_set[1]],[v_set[0],v_set[2]],[v_set[1],v_set[2]]]
        for e in edge_set:
            e_idx = get_edge_idx(mesh,[[e[0].idx(),e[1].idx()]])
            if visited[e_idx] == 0:
                visited[e_idx] = 1
                A = e[0].idx()
                B = e[1].idx()
                BA = [position[A, 0] - position[B, 0], position[A, 1] - position[B, 1]]
                BA_dis = np.linalg.norm(position[B,0:2]-position[A,0:2])
                dir_norm = [length*BA[1]/BA_dis,-length*BA[0]/BA_dis]
                R = []
                R.append((position[A,0]+dir_norm[0],position[A,1]+dir_norm[1]))
                R.append((position[B, 0] + dir_norm[0], position[B, 1] + dir_norm[1]))
                R.append((position[B, 0] - dir_norm[0], position[B, 1] - dir_norm[1]))
                R.append((position[A, 0] - dir_norm[0], position[A, 1] - dir_norm[1]))
                rec = Polygon(R)


                add_weight = 0
                add_area = 0

                face_visited = np.zeros(face_idx.shape[0])

                for nf in mesh.vf(e[0]):
                    face_visited[nf.idx()] = 1
                    nf_index = face_idx[nf.idx()]
                    T = []
                    T.append((position[nf_index[0],0],position[nf_index[0],1]))
                    T.append((position[nf_index[1], 0], position[nf_index[1], 1]))
                    T.append((position[nf_index[2], 0], position[nf_index[2], 1]))
                    tri = Polygon(T)
                    intersection_area = tri.intersection(rec).area
                    add_area += intersection_area
                    add_weight += energy_face[nf.idx()]*intersection_area
                for nf in mesh.vf(e[1]):
                    if face_visited[nf.idx()] == 1:
                        continue
                    else:
                        face_visited[nf.idx()] = 1
                    nf_index = face_idx[nf.idx()]
                    T = []
                    T.append((position[nf_index[0], 0], position[nf_index[0], 1]))
                    T.append((position[nf_index[1], 0], position[nf_index[1], 1]))
                    T.append((position[nf_index[2], 0], position[nf_index[2], 1]))
                    tri = Polygon(T)
                    intersection_area = tri.intersection(rec).area
                    add_area += intersection_area
                    add_weight += energy_face[nf.idx()] * intersection_area
                area[e_idx] = add_area
                weight[e_idx] = add_weight
    return area,weight


def add_fh_face(energy_face,fh_set,mesh_3D):
    face_idx = mesh_3D.fv_indices()
    max_value = np.max(energy_face)
    fh_second_set =[]
    set = []
    for i in fh_set:
        for j in range(3):
            k = np.where(face_idx[:,j]==i)[0]
            for m in k:
                fh_second_set.append(face_idx[k,0])
                fh_second_set.append(face_idx[k, 1])
                fh_second_set.append(face_idx[k, 2])
    for i in fh_second_set:
        for j in range(3):
            k = np.where(face_idx[:,j]==i)[0]
            for m in k:
                energy_face[m] = 10*max_value
                set.append(m)
    return energy_face,set

def maya_get_fh():

    maya_str = [u'D:Mesh.vtx[5258:5415]', u'D:Mesh.vtx[5418:5580]', u'D:Mesh.vtx[5416]', u'D:Mesh.vtx[5417]']
    maya_str.extend(
        [u'D:Mesh.vtx[17553]', u'D:Mesh.vtx[17552]', u'D:Mesh.vtx[17376:17551]', u'D:Mesh.vtx[17554:17698]'])
    maya_str.extend([u'D:Mesh.vtx[181]', u'D:Mesh.vtx[182]', u'D:Mesh.vtx[0:180]', u'D:Mesh.vtx[183:326]'])
    maya_str.extend([u'D:Mesh.vtx[9835]', u'D:Mesh.vtx[9836]', u'D:Mesh.vtx[9771:9834]', u'D:Mesh.vtx[9837:9925]'])
    maya_str.extend(
        [u'D:Mesh.vtx[12286]', u'D:Mesh.vtx[12285]', u'D:Mesh.vtx[12137:12284]', u'D:Mesh.vtx[12287:12463]'])
    maya_str.extend(
        [u'D:Mesh.vtx[21942]', u'D:Mesh.vtx[21943]', u'D:Mesh.vtx[21904:21941]', u'D:Mesh.vtx[21944:22058]'])
    raw_list = maya_str
    raw_index_set = []
    for u in raw_list:
        index = u[u.find('['):]
        if (index.find(':') != -1):
            left = int(index[1:index.find(':')])
            right = int(index[index.find(':') + 1:-1])
            for i in range(left, right):
                raw_index_set.append(i)
        else:
            raw_index_set.append(int(index[1:-1]))

    corr2D_3D = corr2Dto3D(model_3D_path,
                           model_2D_path)
    fh_3D = []
    for point in raw_index_set:
        fh_3D.append(corr2D_3D[point])
    fh_3D = set(fh_3D)
    fh_3D = list(fh_3D)
    print(fh_3D)
    return raw_index_set, fh_3D





def corr2Dto3D(path_3D,path_2D):
    mesh_3D = om.read_trimesh(path_3D)
    faces = mesh_3D.fv_indices()
    mesh_2D = om.read_trimesh(path_2D)
    faces_2 = mesh_2D.fv_indices()
    corr_2Dto3D = {}
    for i in range(faces.shape[0]):
        for j in range(3):
            corr_2Dto3D[faces_2[i,j]]=faces[i,j]
    return corr_2Dto3D


def corr3Dto2D(path_3D,path_2D):

    mesh_3D = om.read_trimesh(path_3D)
    faces = mesh_3D.fv_indices()
    mesh_2D = om.read_trimesh(path_2D)
    faces_2 = mesh_2D.fv_indices()
    corr_3Dto2D = {}
    for i in range(faces.shape[0]):
        for j in range(3):
            if corr_3Dto2D.get(faces[i, j]) == None:
                corr_3Dto2D[faces[i, j]] = [faces_2[i, j]]
            elif len(corr_3Dto2D[faces[i, j]]) >= 1:
                last_num = corr_3Dto2D[faces[i, j]]
                if faces_2[i, j] in last_num:
                    continue
                else:
                    last_num.append(faces_2[i, j])
                    corr_3Dto2D[faces[i, j]] = last_num
    return corr_3Dto2D
def caculate_weight(weight_edge,path_edge):
    cost = 0
    for i in path_edge:
        cost += weight_edge[i]
    return cost


def weight_edge_2D_to_3D(weight_2D):
    model_3D_path = r"./data/raw_model/3D.obj"
    model_2D_path = r"./data/raw_model/2D.obj"
    mesh_2D = om.read_trimesh(model_2D_path)
    mesh_3D = om.read_trimesh(model_3D_path)
    edge_2D = mesh_2D.ev_indices()
    edge_2D = edge_2D[edge_2D[:, 0].argsort()]
    edge_3D = mesh_3D.ev_indices()
    edge_3D = edge_3D[edge_3D[:, 0].argsort()]
    face_2D = mesh_2D.fv_indices()
    face_3D = mesh_3D.fv_indices()
    weight_3D = np.zeros((edge_3D.shape[0]))
    visited_weight_2D = np.zeros(weight_2D.shape[0])
    for i in range(len(face_2D)):
        e_2D_set = [[face_2D[i,0],face_2D[i,1]],[face_2D[i,0],face_2D[i,2]],[face_2D[i,1],face_2D[i,2]]]
        e_3D_set = [[face_3D[i,0],face_3D[i,1]],[face_3D[i,0],face_3D[i,2]],[face_3D[i,1],face_3D[i,2]]]
        for j in range(3):
            e_2D = e_2D_set[j]
            e_3D = e_3D_set[j]
            idx_2D = -1
            idx_3D = -1
            k1 = np.where(edge_2D[:,0]==e_2D[0])[0]
            for k in k1:
                if edge_2D[k,1] == e_2D[1]:
                    idx_2D = k
                    break
            if idx_2D == -1:
                k2 = np.where(edge_2D[:,1]==e_2D[0])[0]
                for k in k2:
                    if edge_2D[k,0] == e_2D[1]:
                        idx_2D = k
                        break
            m1 = np.where(edge_3D[:,0]==e_3D[0] )[0]
            for k in m1:
                if edge_3D[k,1] == e_3D[1]:
                    idx_3D = k
                    break
            m2 = np.where(edge_3D[:,1]==e_3D[0] )[0]
            for k in m2:
                if edge_3D[k, 0] == e_3D[1]:
                    idx_3D = k
                    break
            if visited_weight_2D[idx_2D] == 0:
                visited_weight_2D[idx_2D] = 1
                weight_3D[idx_3D] += weight_2D[idx_2D]
    return weight_3D

if __name__ == "__main__":
    model_2D_path = r"./data/raw_model/2D.obj"
    model_3D_path = r"./data/raw_model/3D.obj"
    action_path = r"./data/action/"

    model_2D_path = r"./data/raw_model/2D.obj"
    model_3D_path = r"./data/raw_model/3D.obj"
    path = r"./data/action/"

    mesh = om.read_trimesh(model_2D_path)
    face_id = mesh.fv_indices()
    for filename in os.listdir(path):
        action_path = path + filename
        print(1)
        energy = np.ones(face_id.shape[0])
        np.save("./weight/energy_face_" + filename + ".npy", energy)

    action_name = ['basketball','dance']
    energy_basketball = calculate_weight_sequence(model_2D_path, action_path + action_name[0], model_3D_path)
    energy_dance = calculate_weight_sequence(model_2D_path, action_path + action_name[1], model_3D_path)
    mesh = om.read_trimesh(model_2D_path)
    energy_face = energy_dance

    # fh_2D,fh_3D = maya_get_fh()
    # energy_face,face_set = add_fh_face(energy_face,fh_2D,mesh)
    np.save("./weight/energy_face_fh.npy",energy_face)

    area,weight = weight_facetoedge_2D(mesh, energy_face)

    weight_3D = weight_edge_2D_to_3D(weight)

    weight_path = "./weight/edge_3D_test.npy"
    np.save(weight_path,weight_3D)


