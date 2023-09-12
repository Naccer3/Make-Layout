from vedo import *
import numpy as np
import matplotlib.pyplot as plt
import openmesh as om
from shapely.geometry import Polygon
import math
from scipy.interpolate import splprep, splev


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

"""
get each edge distance

"""

def calculate_distance(path_3D,path_2D,corr3D_2D):
    mesh = om.read_trimesh(path_3D)
    mesh2 = om.read_trimesh(path_2D)
    edges = mesh.ev_indices()
    edges = edges[edges[:, 0].argsort()]
    list1 = []
    for v in mesh2.vertices():
        list1.append(mesh2.point(v))
    position2 = np.array(list1)
    edge_dis = np.zeros((edges.shape[0]))
    for i in range(edges.shape[0]):
        start_set = corr3D_2D[edges[i,0]]
        end_set = corr3D_2D[edges[i,1]]
        norm_set = []
        for s in start_set:
            for e in end_set:
                norm_set.append(np.linalg.norm(position2[s]-position2[e]))
        edge_dis[i] = min(norm_set)
    return edge_dis

def calculate_distance_3D(path_3D):
    mesh = om.read_trimesh(path_3D)
    edges = mesh.ev_indices()
    edges = edges[edges[:, 0].argsort()]
    list1 = []
    for v in mesh.vertices():
        list1.append(mesh.point(v))
    position2 = np.array(list1)
    edge_dis = np.zeros((edges.shape[0]))
    for i in range(edges.shape[0]):
        s = edges[i,0]
        e = edges[i,1]
        edge_dis[i] = np.linalg.norm(position2[s]-position2[e])
    return edge_dis

"""
Write graph and read layout
"""
# write graph
def WritetoGraph(weight,model_path,graph_path,END= [9822, 9940, 22080, 21371, 7516, 1771, 17332, 5595, 833, 13758]):
    mesh = om.read_trimesh(model_path)
    edges = mesh.ev_indices()
    edges = edges[edges[:,0].argsort()]
    list1 = []
    for v in mesh.vertices():
        list1.append(mesh.point(v))
    position = np.array(list1)

    f = open(graph_path,"w")
    f.writelines("SECTION Graph\n")
    f.writelines("Nodes "+ str(position.shape[0])+"\n")
    f.writelines("Edges "+ str(edges.shape[0])+"\n")
    for i in range(edges.shape[0]):
        f.writelines("E "+str(edges[i,0]+1)+" "+str(edges[i,1]+1)+" "+str(weight[i])+"\n")
    f.writelines("END\n")
    f.writelines("\n")
    f.writelines("SECTION Terminals\n")
    f.writelines("Terminals "+str(len(END))+"\n")
    for i in END:
        f.writelines("T " + str(i+1) + " " + "\n")
    f.writelines("END\n")
    f.writelines("\n")
    f.writelines("EOF")

# calculate layout
def getlayout(graph_path,layout_path):

    alogrithm_path = ".\\algorithm\\steiner_tree-master\\target\\release\\track1.exe" # 顶点从1开始
    os.system(alogrithm_path+" < "+ graph_path + " > " + layout_path)
    with open(layout_path) as f:
        line = f.readlines()
        line = line[1:]
        f = open(layout_path, mode='w', encoding='utf-8')
        f.writelines(line)
        f.close()

# evaluation

def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def circumcenter(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    center_x = ((x1 ** 2 + y1 ** 2) * (y2 - y3) + (x2 ** 2 + y2 ** 2) * (y3 - y1) + (x3 ** 2 + y3 ** 2) * (y1 - y2)) / d
    center_y = ((x1 ** 2 + y1 ** 2) * (x3 - x2) + (x2 ** 2 + y2 ** 2) * (x1 - x3) + (x3 ** 2 + y3 ** 2) * (x2 - x1)) / d
    radius = distance((center_x, center_y), p1)
    return center_x, center_y, radius
#

def get_edge_idx(mesh,path):
    edges = mesh.ev_indices()
    # edges = edges[edges[:, 0].argsort()]
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

def get_neibor_face(face_idx,vertex):
    face_set = []
    face_set.extend(list(np.where(np.array(face_idx)[:,0]==vertex)[0]))
    face_set.extend(list(np.where(np.array(face_idx)[:, 1] == vertex)[0]))
    face_set.extend(list(np.where(np.array(face_idx)[:, 2] == vertex)[0]))
    return face_set



def corr3Dto2D(path_3D,path_2D):

    mesh_3D = om.read_trimesh(path_3D)
    faces = mesh_3D.fv_indices()
    edges = mesh_3D.fe_indices()
    mesh_2D = om.read_trimesh(path_2D)
    faces_2 = mesh_2D.fv_indices()
    edges_2 = mesh_2D.fe_indices()
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


def corr2Dto3D(model_path_3D,model_path_2D):

    mesh_3D = om.read_trimesh(model_path_3D)
    faces = mesh_3D.fv_indices()
    edges = mesh_3D.fe_indices()
    mesh_2D = om.read_trimesh(model_path_2D)
    faces_2 = mesh_2D.fv_indices()
    edges_2 = mesh_2D.fe_indices()
    corr_2Dto3D = {}
    for i in range(faces.shape[0]):
        for j in range(3):
            corr_2Dto3D[faces_2[i,j]]=faces[i,j]
    return corr_2Dto3D


def maya_read_vertex():
    maya_str = [u'D:Mesh.vtx[5258:5415]', u'D:Mesh.vtx[5418:5580]', u'D:Mesh.vtx[5416]', u'D:Mesh.vtx[5417]']
    maya_str.extend([u'D:Mesh.vtx[17553]', u'D:Mesh.vtx[17552]', u'D:Mesh.vtx[17376:17551]', u'D:Mesh.vtx[17554:17698]'])
    maya_str.extend([u'D:Mesh.vtx[181]', u'D:Mesh.vtx[182]', u'D:Mesh.vtx[0:180]', u'D:Mesh.vtx[183:326]'])
    maya_str.extend([u'D:Mesh.vtx[9835]', u'D:Mesh.vtx[9836]', u'D:Mesh.vtx[9771:9834]', u'D:Mesh.vtx[9837:9925]'])
    maya_str.extend([u'D:Mesh.vtx[12286]', u'D:Mesh.vtx[12285]', u'D:Mesh.vtx[12137:12284]', u'D:Mesh.vtx[12287:12463]'])
    maya_str.extend([u'D:Mesh.vtx[21942]', u'D:Mesh.vtx[21943]', u'D:Mesh.vtx[21904:21941]', u'D:Mesh.vtx[21944:22058]'])
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
    return raw_index_set,fh_3D

def add_fh_face(energe_face,fh_set,mesh_3D):
    face_idx = mesh_3D.fv_indices()
    max_value = np.max(energe_face)
    fh_second_set =[]
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
                energe_face[m] = max_value
    return energe_face

def path_3Dto2D(path_3D,
                model_path_3D = r"./data/raw_model/3D.obj",
                model_path_2D = r"./data/raw_model/2D.obj",
                ):
    # 路径为[[s,s1],[s1,s2]]形式
    # 从3D路径变为2D路径
    corr3D_2D = corr3Dto2D(model_path_3D,
                           model_path_2D)
    path = np.loadtxt(path_3D, dtype=int) - 1
    path_2D = []
    for p in path:
        s_set = corr3D_2D[p[0]]
        e_set = corr3D_2D[p[1]]
        newp = [s_set[0],e_set[0]]
        dis = 10000000
        for s in s_set:
            for e in e_set:
                newdis = np.linalg.norm(s-e)
                if newdis < dis:
                    dis = newdis
                    newp = [s,e]
        path_2D.append(newp)
    return path_2D

def find_path_index(path_2D,point_index,visited):
    path_2D = np.array(path_2D)
    tmp1 = np.where(path_2D[:,0]==point_index)[0]
    tmp2 = np.where(path_2D[:,1]==point_index)[0]
    ans = []
    for index in tmp1:
        if visited[index] == 0:
            return index
    for index in tmp2:
        if visited[index] == 0:
            return index

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

def evaluate(model_path_3D = r"./data/raw_model/3D.obj",
            model_path_2D = r"./data/raw_model/2D.obj",
            length=5,
            path_3D="./layout/layout.txt",
            energy_path = './weight/energy_face_fh.npy',
            END = [9822, 9940, 22080, 21371, 7516, 1771, 17332, 5595, 833, 13758]
):
    path_2D = path_3Dto2D(path_3D,
                          model_path_3D,
                          model_path_2D)
    path_2D = np.array(path_2D)
    corr3D_2D = corr3Dto2D(model_path_3D,
                           model_path_2D)
    END_3D = END
    END_2D = []
    for index in END_3D:
        for p in corr3D_2D[index]:
            END_2D.append(p)

    interest_point_set = []
    interest_point_set.extend(END_2D)


    degree = {}
    # 入度加出度超过2的也为端点
    for p in path_2D:
        for i in p:
            if degree.get(i) == None:
                degree[i] = 1
            else:
                degree[i] = degree[i]+1
    for key in degree.keys():
        if degree.get(key) > 2:

            interest_point_set.append(key)

    # 2D 缝合线上的点也为端点
    fh,tmp = maya_read_vertex()
    for key in degree.keys():
        if key in fh :
            interest_point_set.append(key)
    interest_point_set = set(interest_point_set)
    interest_point_set = list(interest_point_set)
    # print(interest_point_set)
    conduct_set = []
    visited = np.zeros(len(path_2D))
    while np.where(visited==0)[0].shape[0] > 0:
        for p in interest_point_set:
            conduct = []
            point = p
            conduct.append(point)
            next_edge = find_path_index(path_2D,point,visited)
            if next_edge == None:
                continue
            visited[next_edge] = 1
            next_point = path_2D[next_edge,0] if path_2D[next_edge,1] == point else path_2D[next_edge,1]
            while next_point != None:
                if next_point in interest_point_set:
                    conduct.append(next_point)
                    conduct_set.append(conduct)
                    break
                else:
                    point = next_point
                    conduct.append(point)
                    next_edge = find_path_index(path_2D, point, visited)
                    visited[next_edge] = 1
                    next_point = path_2D[next_edge,0] if path_2D[next_edge,1] == point else path_2D[next_edge,1]

    # B-spline

    # bezier
    mesh = om.read_trimesh(model_path_2D)
    face_idx = mesh.fv_indices()
    list1 = []
    for v in mesh.vertices():
        list1.append(mesh.point(v))
    position = np.array(list1)
    energy_face = np.load(energy_path)
    all_area = 0
    area = 0
    weight = 0
    poly_set = []
    plt.figure()
    ax = plt.gca()

    # B-spline

    for conduct in conduct_set:
        x = []
        y = []
        neibor_face = []
        for i in conduct:
            x.append(position[i, 0])
            y.append(position[i, 1])
            neibor_face.extend(get_neibor_face(face_idx, i))
        if len(x) <= 3:
            continue

        ss = 10
        kappa = 3 / length
        while np.max(abs(kappa)) > 2 / length:
            if ss > 100000:
                break
            tck, u = splprep([x, y], s=ss, per=False)
            # 使用splev()函数根据得到的参数，分别计算出拟合曲线上100个点的横纵坐标

            u_new = np.linspace(u.min(), u.max(), 1000)
            dx, dy = splev(u_new, tck, der=1)
            d2x, d2y = splev(u_new, tck, der=2)

            # 计算曲线上每个点的速度和加速度
            v = np.sqrt(dx ** 2 + dy ** 2)

            # 计算曲线上每个点的曲率
            kappa = (dx * d2y - dy * d2x) / (v ** 3)
            if np.max(abs(kappa)) > 2 / length:
                ss *= 2

        t = 0.0
        s = 1.0
        l = 0.0
        r = 1.0
        point_pair_set = []
        circle_set = []
        area_add = []

        tck, u = splprep([x, y], s=ss, per=False)
        while l <= 1:
            mid = (l + r) / 2
            x_l, y_l = splev(l, tck)
            x_mid, y_mid = splev(mid, tck)
            x_r, y_r = splev(r, tck)
            A = (x_l, y_l)
            B = (x_mid, y_mid)
            C = (x_r, y_r)
            if abs((B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])) < 0.000001:  # 三点为一条直线
                len_AC = distance(A, C)
                new_len = length / 2
                P1 = [A[0] + new_len / len_AC * (C[1] - A[1]), A[1] + new_len / len_AC * (A[0] - C[0])]
                P2 = [A[0] - new_len / len_AC * (C[1] - A[1]), A[1] - new_len / len_AC * (A[0] - C[0])]
                P3 = [C[0] + new_len / len_AC * (C[1] - A[1]), C[1] + new_len / len_AC * (A[0] - C[0])]
                P4 = [C[0] - new_len / len_AC * (C[1] - A[1]), C[1] - new_len / len_AC * (A[0] - C[0])]
                area_add.append(np.array([P1, P2, P3, P4]).reshape((4, 2)))
                l = r
                r = s
                if l == s:
                    break
            else:
                x_0, y_0, radius = circumcenter(A, B, C)
                evaluate_point_1 = splev((l + mid) / 2, tck)
                evaluate_point_2 = splev((mid + r) / 2, tck)
                evaluate_value_1 = abs(distance((evaluate_point_1[0], evaluate_point_1[1]), (x_0, y_0)) - radius)
                evaluate_value_2 = abs(distance((evaluate_point_2[0], evaluate_point_2[1]), (x_0, y_0)) - radius)
                if evaluate_value_1 > 0.1 or evaluate_value_2 > 0.1:  # error为评估点到圆的距离，大于error则进一步二分查找
                    r = mid
                else:
                    point_pair_set.append((A, C))
                    circle_set.append((x_0, y_0, radius))
                    O = (x_0, y_0)
                    if radius < length / 2:
                        print("error", l, r)
                        OA = [A[0] - O[0], A[1] - O[1]]
                        OC = [C[0] - O[0], C[1] - O[1]]
                        raw_length = radius
                        new_length_1 = 0.001
                        new_length_2 = radius + 0.5 * length
                        P1 = [O[0] + new_length_1 / raw_length * OA[0], O[1] + new_length_1 / raw_length * OA[1]]
                        P2 = [O[0] + new_length_2 / raw_length * OA[0], O[1] + new_length_2 / raw_length * OA[1]]
                        P3 = [O[0] + new_length_2 / raw_length * OC[0], O[1] + new_length_2 / raw_length * OC[1]]
                        P4 = [O[0] + new_length_1 / raw_length * OC[0], O[1] + new_length_1 / raw_length * OC[1]]
                        area_add.append(np.array([P1, P2, P3, P4]).reshape((4, 2)))
                    else:
                        OA = [A[0] - O[0], A[1] - O[1]]
                        OC = [C[0] - O[0], C[1] - O[1]]
                        raw_length = radius
                        new_length_1 = radius - 0.5 * length
                        new_length_2 = radius + 0.5 * length
                        P1 = [O[0] + new_length_1 / raw_length * OA[0], O[1] + new_length_1 / raw_length * OA[1]]
                        P2 = [O[0] + new_length_2 / raw_length * OA[0], O[1] + new_length_2 / raw_length * OA[1]]
                        P3 = [O[0] + new_length_2 / raw_length * OC[0], O[1] + new_length_2 / raw_length * OC[1]]
                        P4 = [O[0] + new_length_1 / raw_length * OC[0], O[1] + new_length_1 / raw_length * OC[1]]
                        area_add.append(np.array([P1, P2, P3, P4]).reshape((4, 2)))
                    l = r
                    r = s
                    if l == s:
                        break
        u_new = np.linspace(u.min(), u.max(), 100)
        x_new, y_new = splev(u_new, tck)
        # ax.plot(x, y, 'o', c = 'red')
        ax.plot(x_new, y_new, c='blue')
        # for circle in circle_set:
        #     cir = plt.Circle((circle[0][0],circle[1][0]),circle[2],fill=False)
        #     ax.add_artist(cir)
        for poly in area_add:
            plt.plot(poly[:, 0], poly[:, 1], c='red')
            plt.plot(poly[[3, 0], 0], poly[[3, 0], 1], c='red')
        poly_set.append(area_add)
        neibor_face = set(neibor_face)
        for index in neibor_face:
            nf_index = face_idx[index]
            T = []
            T.append((position[nf_index[0], 0], position[nf_index[0], 1]))
            T.append((position[nf_index[1], 0], position[nf_index[1], 1]))
            T.append((position[nf_index[2], 0], position[nf_index[2], 1]))
            tri = Polygon(T)
            for poly in area_add:
                R = []
                R.append((poly[0, 0], poly[0, 1]))
                R.append((poly[1, 0], poly[1, 1]))
                R.append((poly[2, 0], poly[2, 1]))
                R.append((poly[3, 0], poly[3, 1]))
                rec = Polygon(R)
                all_area += rec.area / len(neibor_face)
                intersection_area = tri.intersection(rec).area
                area += intersection_area

                weight += intersection_area * energy_face[index]
        # for point_pair in point_pair_set:
        #     plt.scatter(point_pair[0][0],point_pair[0][1],c='red')
        #     plt.scatter(point_pair[1][0], point_pair[1][1], c='red')

    ax.set_aspect('equal', adjustable='box')
    plt.legend()
    plt.show()
    print("INSECT AREA", area)
    print("WEIGHT", weight)
    print("ALL AREA", all_area)


def get_edge_idx(mesh,path):
    edges = mesh.ev_indices()
    edges = edges[edges[:, 0].argsort()]
    edge_idx = np.zeros(len(path),dtype=int)
    for j in range(len(path)):
        for i in range(edges.shape[0]):
            if path[j,0] == edges[i,0]:
                if path[j,1] == edges[i,1]:
                    edge_idx[j] = i
            elif path[j,0] == edges[i,1] :
                if path[j,1] == edges[i,0]:
                    edge_idx[j] = i
    return edge_idx

def caculate_weight(weight_edge,path_edge):
    cost = 0
    for i in path_edge:
        cost += weight_edge[i]
    return cost


if __name__ == "__main__":
    weight_path = "./weight/edge_3D_fhtest.npy"
    model_3D_path = r"./data/raw_model/3D.obj"
    model_2D_path = r"./data/raw_model/2D.obj"
    weight = np.load(weight_path)
    #  #raw layout
    graph_path = "./result/graph/graph.gr"  # Output graph path
    WritetoGraph(np.array(100 * weight, dtype=int), model_3D_path, graph_path)
    layout_path = "./result/layout/layout.txt " # Output layout path
    getlayout(graph_path, layout_path)
    evaluate(path_3D=layout_path)
