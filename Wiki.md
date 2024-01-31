# NeuCoNS- Neuromorphic Communication Network Simulator

Version 0.4.1  
 
---

# Description

NeuCoNS is a Python based simulator to allow the evaluation of different communication network architectures and protocols of Neuromorphic Computing (NC) systems at a high level. This allows to make a well-considered decision for an intended design of the physical implementation in an early design phase of a Project. At such an early stage, the performance of a chosen communication infrastructure is of primary importance, while the actual behavior of the (Spiking) Neural Network (SNN) or individual neurons may be less relevant. Therefore, to simplify the model in favor of large scale network simulations, the nodes internal behaviour such as neuronal membrane potentials, synaptic behaviours, etc. are not considered and only the movement of a spike from the source node to its destinations is modelled. After the spike packet reaches its destination(s), it is dropped.
 
More information on the concept and the implementation of the simulator can be found in *Network Simulator.pdf* in this repository. Initial results obtained using the simulator are presented in [[1](#1), [2](#2)] as well as a validation of the simulator against actual NC hardware in [[3]](#3).

#### Version Description
This is the first open source version of the simulator. The code has been cleaned up in an attempt to make it easy to read, but might still contain leftover code blocks as well as unconventional variable and method names. One of the NN generation types described in [[1](#1)] & [[2](#2)], *Location Based Connectivity* has also been removed (for now) as the current implementation showed less potential and caused some problems. In version 0.4.0, some methods were added specifically for the validation performed in [[3]](#3), these have been removed from version 0.4.1. (will be added as patch)

## Usage

### Setting up the Simulator

As the simulator is implemented in Python, no installation is required. The only thing needed to run the simulator is a Python Interpreter for Python version 3.7 or higher, the python files *network_sim.py*, *netlist_generation.py*, *hardware_graph.py* and *sim_log.py*, and the config file *config.ini*.  
  
Additionally, a netlist file or a connectivity matrix file is required unless one of the implemented test cases is used. More information on the implemented test cases can be found in *Network Simulator.pdf*. Two exemplary matrix files are also included in this repository and can be found in the folder matrix_files.


### Running the Simulator

By running the command:  
> python3 network_sim.py *<Run_Name> <nr_processors>*  

a simulation is started and the results are stored in a directory *<Run_name>*. Also included in this directory are a *log*-file, a copy of the *config.ini* file and, for certain placement algorithms, files defining the neuron placement. The simulator will read out the parameters to use for this run from the config file located in the current working directory. Thus, to run the simulator, the config file has to be located in the current working directory. The Python files on the other hand, can be located in a different directory. However, this does require the addition of the path to the *network_sim.py* file in the command.  

The second argument *<nr_processors>* is optional and can be used to speed up the simulation by parallelizing it over the given number of cores. By default, only a single core is used in case no argument is given.

If a directory with the given *<Run_Name>* already exists in the current working directory, the user is asked whether the existing folder should be overwritten. In this case a new run is started with the same name and the results are overwritten. The original *log*-file on the other hand, is preserved and the name of the new *log*-file is extended with the date and time the run was started. This *log*-file stores information regarding the simulation run, such as run time, run settings, notices, warnings and errors.  

### The Config File

The config file is used to set the parameters of the simulation. The file is divided in a couple of sections, each covering the respective settings. Some of these settings are explained in detail in *Network Simulator.pdf*. 
- Simulation Settings
	- ignore unused:  
	True/False; if true, nodes & links which are unused, i.e. have 0 packets passing through them, are omitted during the calculation of the traffic statistics.
	- plot heatmaps:  
	True/False; if true, heatmaps are plotted automatically for every run and are stored in the subdirectory *graphs*. The file name for every plot is generated automatically according to the parameters of the current run. Because of this, the file names and graph titles might become cluttered, especially when multiple parameters are set with loop values.  
	Attention! For large (hardware) network sizes, the link heatmap becomes hard to read.
	- loop variable:  
	Specify a loop variable; the simulation will automatically run multiple times, once for each value specified in the *loop value* parameter. Leave empty to turn this function off. The following variables can be looped in this version:  
	[*testcase arguments, mapping arguments, neurons per node, link delay, router delay*]  
	An additional option that can be set here is *repeat*. In this case the simulation will run multiple times with the same parameter values. This is usefull to determine statistics in case that randomness is included anywhere in the simulator, such as the NN generation (not compatible with the netlist based NN generation). In this case the loop value is a list of identifiers for each run, can be arbitrary names or numbers.
	- loop values:  
	List of values separated by "," to be used for the loop variable.
- Neural Network
	- NN generation;  
	Netlist/Matrix, a more detailed description of both of these types of NN generation can be found in the documentaion. Matrix based generation has performance benefits over the netlist based generation and requires (much) less working memory. Netlist based generation on the other hand offers the potential for more complex connectivity schemes and stores the exact connectivity data for later use or the use in other simulators.
	- testcase arguments:  
	A dictionary containing the key: *testcase_type*, which specifies the type of netlist to be generated, as well as entries for all input parameters of the selected testcase. See the documentation for a list of the input parameters of each testcase. In case a (user defined) netlist file has to be read out, *testcase_type* should be set to *file* and a second entry *file_name* should be added. 
	- connectivity matrix file:  
	Defines the file name of the matrix file to be used; should also include the path to this matrix file in relation to the current working directory.
	- FR defined  
	True/False; if true, the third column in the matrix file is interpreted as the firing rate of the corresponding population. The matrix file in this case should have an additional column compared to the matrix file that uses used FR defined := False.
- Communication Protocol
	- routing:
	Sets the routing algorithm(s) to be used for the simulation. List input separated by "," possible.  
	Included by default (depending on the topology selected): [*dimension order routing (DOR), longest dimension first routing (LDFR), shortest path routing/dijkstra (SPR), enhanced shortest path routing (ESPR), neighbour exploration routing (NER)*]
	- casting:
	Sets the casting type(s) to be used for the simulation. List input separated by "," possible.  
	Included by default: [*Unicast (UC), Local_Multicast (LMC), Multicast (MC), Broadcast (BC), Broadcastfirst-UC/LMC/MC (BCF-UC/LMC/MC)* [[4]](#4)]
- Neuron Mapping
	- algorithm:  
	Sets the mapping algorithm used; different options available depending on the NN generation type selected.  
	Netlist based: [*random, sequentially*]  
	Matrix based: [*random, sequentially, population_grouping, area_grouping*]
	- neurons per node:  
	Set the number of neurons mapped per node.
	- mapping parameters:
	A dictionary containing the input parameters of the selected mapping algorithm; **not used with the currently included mapping algorithms**
- Network Topology
	- topology:
	Defines the network topology(s) to be analysed in the simulation. List input separated by "," possible.  
	Includes by default: [*Mesh4, Mesh6, Mesh8, Mesh3D, SpiNNaker, TrueNorth, BrainScaleS, MultiMesh[8-gridsizes][6-gridsizes][4-gridsizes]* [[4]](#4)]  
	- torus:
	True/False; if true, wrap around connection are created for the mesh networks, connecting left and right edges as well as the lower and upper edges of the network to each other.
	- router delay:
	Sets the time required for a packet to pass through a router (node). By default set to 1, leading to a latency estimation in number of "hops".
	- link delay:
	Sets the time required to pass a link with length 1. The link delay is multiplied with the link length in case of a physically long connection. Defaults to 0 for a latency estiamtion in terms of "hops".

All text inputs in the config file are case ***in***sensitive. Settings which don't apply in combination with other settings, are automatically neglected and don't need to be deleted. Not all options are listed for every setting, especially lesser known options which require more explanation are omitted. Effort was spent to make the simulator work for fully qualified parameter names as well as abbreviations and potential alternative names. However, this will not be foolproof and thus might lead to unexpected errors if faulty settings are selected.

## Simulation results

After a simulation has finished (successfully), the results can be found in the run directory. These results consist of the automatically plotted heatmaps (when this option is set to true) in the graphs subdirectory, <Run_name>_latency.json and <Run_name>_traffic.json  files, and a  <Run_name>_summary.csv file.  
The *json*-files store all the network traffic data in nested dictionaries. The upper levels of these dictionaries are the same for both these files as they are initialized corresponding to the different runs executed in the simulation and follow the following hierarchy: Loop value, topology, mapping algorithm, routing algorithm, casting type. However, the nested dictionary is only created in case the corresponding simulator parameter is given as list input. In case the input parameter is a single value for a simulation, the corresponding layer in the nested dictionary is omitted. The final dictionary has a nested dictionary for each individual run executed by the simulator.  
From this point, the two *json*-files differ.  
In the latency dictionary, the nested dictionary for an individual run looks as follows:
  
`{"per_neuron": {<Neuron ID>: Latency}, "average": xx.xx, "min": xx.xx, "max": xx.xx, "median": xx.xx}`

The "per_neuron" nested dictionary contains an entry for each neuron in the test case allong with the maximum latency/distance a spike generated by that neuron had to travel.  
The structure of the traffic dictionary is a little bit more complex:  
```
{
<Node ID>: {"int_packets_handled": xx.xx, "ext_packets_handled": xx.xx, "edges":{<Node ID>: {"length": x, "packets_handled": xx.xx}}},
"int_packets_handled": {"average": xx.xx, "min": xx.xx, "max": xx.xx, "median": xx.xx},
"ext_packets_handled": {"average": xx.xx, "min": xx.xx, "max": xx.xx, "median": xx.xx},
"packets_handled_per_node": {"average": xx.xx, "min": xx.xx, "max": xx.xx, "median": xx.xx},
"spikes_per_link": {"average": xx.xx, "min": xx.xx, "max": xx.xx, "median": xx.xx},
}
```  

Each node in the hardware graph is included in this dictionary, even if no spikes go through them. The nested *edges*  dictionary contains all the nodes the node is connected to, representing the uni-directional link between the two nodes.

To make the results easier to read (for a human), the data is summarized in the *csv*-file. Here, the data is compressed into averages and min-/max- values for the different metrics and stored in a manner that can be read out in a table format by programs such as Excel. Please note that the *csv*-file uses ";" as the delimiter as the comma is used as decimal separator.  

## Adding to the Code

The code has been structured in a way that different topologies, castingtypes, routing algorithms, mapping algorithms and testcases can be added with ease.  
New testcases can be added either by adding a netlist- or a matrix file, or by adding a netlist generation method to the module *netlist_generation.py* . The new netlist generation method should return a netlist dictionary and the path to where this netlist is saved as a *.json*-file. The format of the netlist dictionary and the netlist file is described in the documentation that can be found in this repository, as well as the format required for the matrix file. After implementing a new netlist generation method, the option for the new type of testcase should also be added to the factory method in the same module.

New routing or mapping algorithms can be added by implementing the algorithm in the *Network* class or corresponding child class (in case of a topology specific routing algorithm) in the *hardware_graph.py* module. Again, the new method should also be added to the routing/mapping factory method of that class. The routing algorithm (in case of the regular meshes) receives a source-node ID and a list of destination nodes and returns 2 dictionaries. One path dictionary, as explained in the documentation, and one distance dictionary in which the destination nodes are the keys and the calculated distance to the nodes the corresponding values. The mapping algorithms receive the number of neurons allowed per node, either the netlist dictionary or the connectivity matrix, and a *rest_args* dictionary to facilitate the passing of other parameters set with the *mapping parameters* dictionary. The *aux_args* dictionary is used to pass a keyword addressed argument list to be used by more complex mapping algorithms. The mapping method assigns either neuron IDs - in case of the netlist based simulation - or a tuple identifying the number of neurons and the population name (<nr_neurons>, <population>) (in case of the matrix based simulation) to the node object's neuron list. The mapping algorithm does not return anything.

Adding a new casting type is done in a way similar to the routing- and mapping- algorithms. However, while the casting type has to be implemented in the corresponding (parental) class, the method has to be added to the casting factory method located in the *network_sim.py* module. The casting methods receive the Node_ID and the node object of the node that is being processed, along with a routing_type and either a netlist or matrix dictionary. The casting method determines all the destinations of all neurons located on that node, calculates the resulting routes to all these destinations and potentially merges them according to the casting type. It then returns the Node_ID together with a dictionary listing each link used by neurons from that node and the corresponding frequency the link is used, the number of spikes generated in that node and the latency dictionary for all neurons in that node.

Finally, a new topology can be added by creating a class object - potentially as child class of an already existing topology - for the corresponding topology in the *hardware_graph.py* module. Again, the option to choose this topology has to be added to the graph factory method in the  *hardware_graph.py* module. The new class should define a *create_network* method which adds the nodes and links representing the desired topology to the graph. Additionally, different routing, mapping and casting methods can be added to the class accordingly.

To make the factory methods easy to find, they have all been clearly marked with comments and are located conveniently at the top of the module (when possible). 
The class specific factory methods are class methods and thus have to be located within the class object.

---

## Known issues

The automatic naming of the *<placement file>* and heatmap plots becomes cluttered in case multiple input variables are defined as lists in inputs. The automatically plotted heatmap of the traffic per link has some additional limitations as well. It doesn't scale well for large hardware graph sizes and can't properly visualize the torus connections in a 2D plot. 

---

## References

<a id="1">[1]</a> R. Kleijnen, M. Robens, M. Schiek and S. van Waasen, "A Network Simulator for the Estimation of Bandwidth Load and Latency Created by Heterogeneous Spiking Neural Networks on Neuromorphic Computing Communication Networks," *2021 IEEE 14th International Symposium on Embedded Multicore/Many-core Systems-on-Chip (MCSoC)*, 2021, pp. 320-327, doi: 10.1109/MCSoC51149.2021.00054.  
<a id="2">[2]</a> R. Kleijnen, M. Robens, M. Schiek, and S. van Waasen, “A Network Simulator for the Estimation of Bandwidth Load and Latency Created by Heterogeneous Spiking Neural Networks on Neuromorphic Computing Communication Networks,” *Journal of Low Power Electronics and Applications*, vol. 12, no. 2, p. 23, Apr. 2022, doi: 10.3390/jlpea12020023.  
<a id="3">[3]</a>  R. Kleijnen, M. Robens, M. Schiek and S. van Waasen "Verification of a neuromorphic computing network simulator using experimental traffic data. Front. Neurosci. 16:958343, Juli 2022, doi: 10.3389/fnins.2022.958343 
<a id="4">[4]</a> K. Kauth, T. Stadtmann, R. Brandhofer, V. Sobhani and T. Gemmeke, "Communication Architecture Enabling 100x Accelerated Simulation of Biological Neural Networks," *2020 ACM/IEEE International Workshop on System Level Interconnect Prediction (SLIP)*, 2020, pp. 1-8.