# Make-Layout
Code for "Computational Design of Wiring Layout on Tight Suits with Minimal Motion"

This is an algorithm for generating intelligent clothing wire layout. It mainly consists of two parts: calculating weight (get_edge_weight.py) and calculating layout (utils.py). The function of calculating layout uses the Steiner tree problem algorithm provided by Yoichi. 
# Required
Python 3.7

Rust
# Project structure
```
-get_layout
	-data
		-action : Used to store sports cloth obj sequences
		-raw_model ： Used to store the initial state of the fabric (2D and 3D patterns)
	-result
		-graph ： Data used to store graphs in STP problems
		-layout ： Used to store layout results
	-steiner_tree-master ： STP algorithm module
	-weight ：Used to store weights
	get_edge_weight.py ：Calculate weight
	utils.py ： Calculate layout
	visual.py : Layout visualization
```
# How to build
Mainly required to create STP solver Specific method: 
Run the command line in the path where steiner_tree-master is located:
```PowerShell
$ cargo build --release
```
# How to use
## Quick Start
If you just want to quickly start viewing the project: 

Run utils.py to get the layout and evaluation results on the 2D pattern

Run visual.py to get the visualization results on the 3D pattern.

Run from command line:
```
PowerShell
> python utils.py
> python visual.py
```
## Detail
1.Modify the sensor node position: WritetoGraph function END variable control, fill in the sensor node list (vertex serial number list in 3D obj), the default is:
	

```Python
WritetoGraph(END=[])
```
2.Calculate personalized weight:
If you need to change the clothing and applicable actions, place the original 2D and 3D patterns in the folder data\raw_model, store the clothing movement obj sequence in the folder (for example, named myaction) and place it in data\action.
```Python
model_2D_path = r"./data/raw_model/2D.obj"
model_3D_path = r"./data/raw_model/3D.obj"
action_path = r"./data/action/myaction"
FACE ENERGY
energy_myaction = calculate_weight_sequence(model_2D_path, action_path, model_3D_path)
EDGE WEIGHT
mesh = om.read_trimesh(model_2D_path)
area,weight = weight_facetoedge_2D(mesh, energy_face)
weight_3D = weight_edge_2D_to_3D(weight) # FINAL WEIGHT
weight_path = "./weight/edge_3D_test.npy"
np.save(weight_path,weight_3D)
```
After using personalized weights, calculating the layout requires modifying the sensor vertex position (refer to 1)
3.Evaluation: Allows changing the width of the wire (length parameter control), modifying the weight (energy_path control)
