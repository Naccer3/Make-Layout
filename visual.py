from vedo import *
import colorcet  # https://colorcet.holoviz.org
import numpy as np
import openmesh as om


# "depth peeling" may improve the rendering of transparent objects
settings.useDepthPeeling = True
settings.multiSamples = 0  # needed on OSX vtk9


weight_path = "./weight/energe_face_Air_Squat_Bent_Arms.npy"
model_3D_path = r"./data/raw_model/3D.obj"
model_2D_path = r"./data/raw_model/2D.obj"

mesh = om.read_trimesh(model_3D_path)
edges = mesh.ev_indices()
edges = edges[edges[:,0].argsort()]
faces = mesh.fv_indices()
list1 = []
for v in mesh.vertices():
    list1.append(mesh.point(v))
position = np.array(list1)

man = Mesh([position,faces])

scals = np.load(weight_path)
max,min = np.max(scals),np.min(scals)
tmp = []
for i in scals:
    tmp.append(log(i)/log(max))
scals = tmp


mycmap = colorcet.rainbow4
alphas = np.linspace(1, 1, num=len(mycmap))
man.cmap(mycmap, scals,on ='cells' ).addScalarBar()

plt = Plotter(N=2,axes=5).addShadows().render()
plt.at(0).show(man, __doc__, viewup="z", axes=5)
plt.at(1).show(man, __doc__, viewup="z", axes=5)
path = np.loadtxt("./result/layout/layout_Xsens_stre.txt",dtype=int)-1

END = [10687, 11900, 22907, 22790, 13397, 4474, 15345, 3787, 14017, 9752, 19033, 19661, 7197, 17090, 5429]
for i in END:
    point = Point((position[i,0],position[i,1],position[i,2]),c='red')
    plt.at(0).show(__doc__,point)

for path1 in path:
    tube = Tube(position[path1,:],c="red", r=10)
    plt.at(1).show(__doc__, tube)
plt.interactive().close()

