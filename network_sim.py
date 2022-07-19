"""
network_sim.py

Copyright (C) 2022, R. Kleijnen, Forschungszentrum Jülich, Central Institute of Engineering, Electronics and Analytics—Electronic Systems (ZEA-2)

This file is part of NeuCoNS.
NeuCoNS is a free application: you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

NeuCoNS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with NeuCoNS. If not, see <https://www.gnu.org/licenses/>.
"""

import sys
import time
import os
import netlist_generation
import hardware_graph
import json
import sim_log
from sim_log import RoutingError, MappingError, CastingError
import math
import statistics
import configparser
import concurrent.futures
import operator
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

# Set the global parameter defaults
GLOBAL_VAR = {
	'nr_cores': 1
}

# Define Colors and Colormap for automatic plotting functions
turbo = cm.get_cmap('viridis', 256)
newcolors = turbo(np.linspace(0, 1, 256))
idle_color = np.array([0.7, 0.7, 0.7, 1])
newcolors[0, :] = idle_color
newcmp = ListedColormap(newcolors)

CD_colors = ['#023d6b', '#eb5f73', '#b9d25f', '#af82b9', '#ebebeb', '#adbde3', '#faeb5a', '#fab45a']
cm = 1 / 2.54


def createFolder(folder_name):
	directory = os.getcwd() + '/' + folder_name
	try:
		if not os.path.exists(directory):
			os.makedirs(directory)
			print('Directory ' + directory + ' created.')
	except OSError:
		print('Error: Creating directory. ' + directory)


def string_to_tuple(str_tuple):
	if str_tuple == '':
		return
	else:
		return tuple(int(x) for x in str_tuple.strip("()").split(","))


def colormap_value(value, max_value):
	x = value / max_value
	return newcmp(x)[:3]


########################################################################################################################
# Factory Methods
########################################################################################################################
def graph_factory_function(topology, nr_nodes):
	if topology.lower() == 'mesh4' or topology.lower() == 'square':
		return hardware_graph.Mesh4, math.ceil(math.sqrt(nr_nodes))
	elif topology.lower() == 'mesh6' or topology.lower() == 'triangular':
		return hardware_graph.Mesh6, math.ceil(math.sqrt(nr_nodes))
	elif topology.lower() == 'mesh8':
		return hardware_graph.Mesh8, math.ceil(math.sqrt(nr_nodes))
	elif topology.lower() == 'truenorth':
		return hardware_graph.TrueNorth, math.ceil(math.sqrt(nr_nodes))
	elif topology.lower() == 'spinnaker':
		nr_boards = math.ceil(nr_nodes / 48)
		return hardware_graph.SpiNNaker, nr_boards
	elif topology.lower().startswith('multi'):
		size = math.ceil(math.sqrt(nr_nodes))
		Meshes = topology[10:-1].split('][')
		mesh8 = []
		mesh6 = []
		mesh4 = []

		if Meshes[0]:
			for i in Meshes[0].split(';'):
				length = int(i)
				if length > (size / 2):
					sim_log.notice(
						f'Higher level long range connections of length {length} '
						f'are omitted as this is larger than the largest distance between two points in the network.')
				else:
					mesh8.append(length)
		if Meshes[1]:
			for i in Meshes[1].split(';'):
				length = int(i)
				if length > (size / 2):
					sim_log.notice(
						f'Higher level long range connections of length {length} '
						f'are omitted as this is larger than the largest distance between two points in the network.')
				else:
					mesh6.append(length)
		if Meshes[2]:
			for i in Meshes[2].split(';'):
				length = int(i)
				if length > (size / 2):
					sim_log.notice(
						f'Higher level long range connections of length {length} '
						f'are omitted as this is larger than the largest distance between two points in the network.')
				else:
					mesh4.append(length)
		longest_range = max(mesh4 + mesh6 + mesh8)
		size = longest_range * math.ceil(size / longest_range)

		return hardware_graph.MultiMesh, size, mesh8, mesh6, mesh4
	elif topology.lower() == 'mesh3d' or topology.lower() == 'cube':
		return hardware_graph.Mesh3D, int(math.ceil(nr_nodes ** (1 / 3)))
	elif topology.lower().startswith('hubnetwork-bc'):
		N0 = int(topology[13:])
		size = math.ceil(math.ceil(math.sqrt(nr_nodes)) / N0) * N0
		return hardware_graph.HubNetwork_BC, size, N0
	elif topology.lower().startswith('hubnetwork'):
		linklength = int(topology[10:])
		size = math.ceil(math.ceil(math.sqrt(nr_nodes)) / linklength) * linklength
		return hardware_graph.HubNetwork, size, linklength
	else:
		return None, 0


def casting_factory_function(network, casting_type):
	casting_type = casting_type.lower()
	topology = network.type
	if (topology == 'TrueNorth' and casting_type != 'lmc') or \
		(topology in ['SpiNNaker', 'BrainscaleS'] and casting_type != 'mc'):
		raise CastingError

	if casting_type == 'bc' or casting_type == 'broadcast':
		return network.broadcast
	elif casting_type == 'uc' or casting_type == 'unicast':
		return network.unicast
	elif casting_type == 'lmc' or casting_type == 'local_mc' or casting_type == 'local_multicast':
		return network.local_multicast
	elif casting_type == 'mc' or casting_type == 'multicast':
		return network.multicast
	elif casting_type.startswith('bcf-'):
		if casting_type == 'bcf-uc':
			return network.broadcastfirst_unicast
		elif casting_type == 'bcf-lmc':
			return network.broadcastfirst_local_multicast
		elif casting_type == 'bcf-mc':
			return network.broadcastfirst_multicast
	elif casting_type.endswith('flood'):
		if casting_type == 'uc-flood':
			return network.uc_flood
		elif casting_type == 'mc-flood':
			return network.mc_flood
	else:
		raise CastingError


########################################################################################################################
# Main Function
########################################################################################################################
def main():
	try:
		run_name = sys.argv[1]
	except IndexError:
		print(
			f'No run name defined.\n'
			f'Module "network_sim.py" takes one argument, none where given.\n'
			r'Abort run.')
		sys.exit()

	if os.path.isfile('config.ini'):
		read_config = configparser.ConfigParser()
		read_config.read('config.ini')
	else:
		print(
			f'config.ini file not found.\n'
			f'Make sure the config file is located in {os.getcwd()}')
		sys.exit()

	if os.path.exists(run_name):
		print(f'Simulation folder {run_name} already exists...')
		confirmation = input('Overwrite existing folder? [y/n] ')
		if not confirmation == 'y':
			print('Simulation is terminated.')
			sys.exit()

	createFolder(run_name)
	path = os.getcwd() + '/'
	os.chdir(run_name)
	sim_log.create_log(run_name, read_config)
	start_time = time.time()

	try:
		GLOBAL_VAR['nr_cores'] = int(sys.argv[2])
	except IndexError:
		print(
			f'Using single core only')
		GLOBAL_VAR['nr_cores'] = 1

	# Read out config.ini file
	settings = {
		'run_name': run_name,
		'ignore_unused': True if read_config.get("Sim Settings", "Ignore Unused").lower() == 'true' else False,
		'plot_heatmaps': True if read_config.get("Sim Settings", "Plot Heatmaps").lower() == 'true' else False,
		'path': path
	}

	loop_var = read_config.get("Sim Settings", "Loop Variable").lower()
	if loop_var:
		loop_values = [float(x) for x in read_config.get("Sim Settings", "Loop Values").split(',')]
	else:
		loop_var = None
		loop_values = ['single run']

	settings['routing_types'] = [item.strip() for item in read_config.get("Communication Protocol", "Routing").split(',')]
	settings['casting_types'] = [item.strip() for item in read_config.get("Communication Protocol", "Casting").split(',')]

	settings['mapping_types'] = [item.strip() for item in read_config.get("Neuron Mapping", "algorithm").split(',')]
	settings['neurons_per_node'] = int(read_config.get("Neuron Mapping", "Neurons per Node"))

	settings['mapping_args'] = {}
	for key, value in json.loads(read_config.get("Neuron Mapping", "Mapping Parameters")).items():
		try:
			settings['mapping_args'].update({key: float(value) if float(value) % 1 else int(value)})
		except ValueError:
			settings['mapping_args'].update({key: value})

	settings['topologies'] = [item.strip() for item in read_config.get("Network Topology", "Topology").split(',')]
	if any([top.lower().startswith('hubnetwork') for top in settings['topologies']]) \
		or any([top.lower().startswith('multi') for top in settings['topologies']]):
		settings['secondary'] = True
	else:
		settings['secondary'] = False

	settings['torus'] = \
		[True if item.strip().lower() == 'true' else False
		 for item in read_config.get("Network Topology", "Torus").split(',')]
	settings['hardware_graph.t_link'] = float(read_config.get("Timing", "link delay"))
	settings['hardware_graph.t_router'] = float(read_config.get("Timing", "router delay"))

	hardware_graph.t_link = settings['hardware_graph.t_link']
	hardware_graph.t_router = settings['hardware_graph.t_router']

	results_traffic = {}
	results_latency = {}

	# Select the type of NN used for the simulation.
	run_type = read_config.get("Neural Network", "NN generation")
	if run_type.lower() == 'testcase' or run_type.lower() == 'netlist':
		testcase_dict_template = {}
		for key, value in json.loads(read_config.get("Neural Network", "testcase arguments")).items():
			try:
				testcase_dict_template.update({key: float(value) if float(value) % 1 else int(value)})
			except ValueError:
				testcase_dict_template.update({key: value})
		settings['testcase_type'] = testcase_dict_template['testcase_type']
		del testcase_dict_template['testcase_type']
		settings['testcase_args'] = testcase_dict_template
		createFolder('../netlists')
		createFolder('placement_files')

		sim_function = testcase_analysis

	elif run_type.lower() == 'matrix':
		settings['matrix_file'] = read_config.get("Neural Network", "Connectivity matrix file")
		FR_defined = True if read_config.get("Neural Network", "FR defined").lower() == 'true' else False
		settings['connectivity_matrix'] = \
			netlist_generation.read_connectivity_probability_matrix(settings['matrix_file'], FR_defined)

		settings['testcase_args'] = {}

		# createFolder('placement_files')
		sim_function = probability_matrix_analysis

	else:
		sim_log.fatal_error(
			f'Incorrect NN generation selected - {run_type}, '
			f'Valid options are [testcase, matrix]')
		sim_function = None

	# Set up loop variable
	for step in loop_values:
		if loop_var:
			sim_log.message(
				f'({loop_values.index(step) + 1} / {len(loop_values)}).\n{"-" * 64}\n'
				f'\nStart run {loop_var}: {step}')
			print(
				f'({loop_values.index(step) + 1} / {len(loop_values)}).\n{"-" * 64}\n'
				f'Start run {loop_var}: {step}')
		else:
			sim_log.message(
				f'{"-" * 64}\n'
				f'\nStart simulation (single run)')
			print(
				f'{"-" * 64}\n'
				f'Start simulation (single run)')

		if loop_var in settings['testcase_args'].keys():
			if step % 1:
				settings['testcase_args'].update({loop_var: float(step)})
			else:
				settings['testcase_args'].update({loop_var: int(step)})
		elif loop_var in settings['mapping_args'].keys():
			if step % 1:
				settings['mapping_args'].update({loop_var: float(step)})
			else:
				settings['mapping_args'].update({loop_var: int(step)})
		elif loop_var == 'neurons per node' or loop_var == 'npn':
			settings['neurons_per_node'] = int(step)
			step = int(step)
		elif loop_var == 'router delay':
			hardware_graph.t_router = step
			settings['hardware_graph.t_router'] = step
			hardware_graph.t_router = settings['hardware_graph.t_router']
		elif loop_var == 'link delay':
			hardware_graph.t_link = step
			settings['hardware_graph.t_link'] = step
			hardware_graph.t_link = settings['hardware_graph.t_link']
		elif not loop_var:
			sim_log.notice('No loop variable selected.')
		elif loop_var == 'repeat':
			# Repeat the simulation multiple times with the same variables.
			# Useful to investigate the effect of randomness
			pass
		else:
			sim_log.fatal_error(
				f'The loop variable selected for the simulation: {loop_var} is invalid.')

		# Run simulation for current variable step
		try:
			results_iteration = sim_function(settings, loop_var, step, run_name, loop_values.index(step))
		except sim_log.SimError:
			break

		# Store the results of the current iteration
		if loop_var:
			results_traffic[step] = results_iteration[0]
			results_latency[step] = results_iteration[1]
		else:
			results_traffic = results_iteration[0]
			results_latency = results_iteration[1]

		# Write results to file after each run
		with \
			open(run_name + '_traffic.json', 'w') as traffic_file, \
			open(run_name + '_latency.json', 'w') as latency_file:
			json.dump(results_traffic, traffic_file)
			json.dump(results_latency, latency_file)

	sim_log.end_log(hardware_graph.convert_time(time.time() - start_time))


########################################################################################################################
# Methods setting up the simulation of individual runs for the different types of NN generation
########################################################################################################################
def testcase_analysis(settings, loop_var, step, run_name, i):
	if i == 0:
		# write summary file header
		with open(run_name + '_summary.csv', 'w') as file:
			if loop_var:
				file.write(f'{loop_var};')
			file.write(
				f'Topology;Mapping;Routing;Casting;'
				f'Avg spikes per router;Min spikes per router;Max spikes per router;'
				f'Avg int. spikes per router;Min int. spikes per router;Max int. spikes per router;'
				f'Avg ext. spikes per router;Min ext. spikes per router;Max ext. spikes per router;')
			if settings['secondary']:
				file.write(
					f'Average spikes per link (primary layer);Minimum spikes per link (primary layer);'
					f'Maximum spikes per link (primary layer);'
					f'Average spikes per link (secondary layer);Minimum spikes per link (secondary layer);'
					f'Maximum spikes per link (secondary layer);'
					f'Average latency;Minimum latency;Maximum latency\n')
			else:
				file.write(
					f'Average spikes per link;Minimum spikes per link;Maximum spikes per link;'
					f'Average latency;Minimum latency;Maximum latency\n')

	netlist_generator = netlist_generation.netlist_factory(settings['testcase_type'])
	settings['mapping_args']['path'] = settings['path']

	try:
		netlist, netlist_file = netlist_generator(**settings['testcase_args'], path=settings['path'] + 'netlists/')
		nr_nodes = math.ceil(len(netlist) / settings['neurons_per_node'])
		if nr_nodes == 1:
			sim_log.error(
				'Simulated NC system is a single node, and thus no communication network exists.\n'
				'Continue with next run.')
		elif nr_nodes < 1:
			sim_log.error(
				f'Simulated NC system with {nr_nodes} node(s).\n'
				'Continue with next run.')

	except TypeError:
		netlist, netlist_file, nr_nodes = None, None, None
		sim_log.message('\nFault in netlist dictionary template:')
		for parameter in settings['testcase_args']:
			sim_log.message(f'\t{parameter}: {settings["testcase_args"][parameter]}')
		sim_log.fatal_error('See netlist dictionary template above')

	sim_log.message('Netlist parameters:')
	for parameter in settings['testcase_args']:
		sim_log.message(f'\t{parameter}: {settings["testcase_args"][parameter]}')
	sim_log.message('\n')

	results_traffic = {}
	results_latency = {}

	sim_log.message(
		f'Analyse {len(settings["topologies"]) * len(settings["torus"])} '
		f'different topologie(s) for given neural network.')
	i = 1
	for topology in settings['topologies']:
		for torus_option in settings['torus']:
			if (torus_option and topology.lower() in ['truenorth', 'brainscales']) or \
					(not torus_option and topology.lower() == 'spinnaker'):
				sim_log.notice(
					f'Selected combination of {topology} and {torus_option} is not possible.\n'
					f'Skip...')

			if topology.lower() == 'brainscales':
				network = hardware_graph.BrainscaleS()
			elif topology.lower().startswith('multi'):
				graph_generator, size, grid8, grid6, grid4 = graph_factory_function(topology, nr_nodes)
				network = graph_generator(size, torus_option, grid8, grid6, grid4)
			else:
				graph_generator, size = graph_factory_function(topology, nr_nodes)
				if not graph_generator:
					sim_log.error(f'One of the selected topologies - {topology} - is not defined.')
					continue
				network = graph_generator(size, torus_option)

			results_traffic[topology + '_Torus' * torus_option] = {}
			results_latency[topology + '_Torus' * torus_option] = {}

			sim_log.message(f'\n{"*" * 16}\nTopology {i} of {len(settings["topologies"]) * len(settings["torus"])}')
			print(f'Topology {i} of {len(settings["topologies"]) * len(settings["torus"])}')
			i += 1
			for mapping_type in settings['mapping_types']:
				results_traffic[topology + '_Torus' * torus_option][mapping_type] = {}
				results_latency[topology + '_Torus' * torus_option][mapping_type] = {}
				network.reset_mapping(netlist)
				try:
					mapping = network.mapping_module(mapping_type)
					mapping(settings['neurons_per_node'], netlist, **settings['mapping_args'])
					print(f'Finished mapping of neurons ({mapping_type}).')

					with open(f'placement_files/{mapping_type}', 'w') as file:
							json.dump({neuron: netlist[neuron]['location'] for neuron in netlist.keys()}, file)

				except MappingError:
					sim_log.message(f'Mapping failed (see error above), abort run.\n')
					continue
				except KeyError:
					sim_log.fatal_error(f'One of the neurons in the netlist is not mapped to a physical location.')

				T0 = time.time()
				for routing_type in settings['routing_types']:
					results_traffic[topology + '_Torus' * torus_option][mapping_type][routing_type] = {}
					results_latency[topology + '_Torus' * torus_option][mapping_type][routing_type] = {}
					try:
						print("\r\033[K", end='')
						for casting_type in settings['casting_types']:
							try:
								T1 = time.time()
								network.reset_traffic()

								latency_run = {}
								nodes_processed = 0
								ratio = 0
								process_Node = casting_factory_function(network, casting_type)
								for Node_ID, Node_obj in network.nodes.items():
									if Node_obj.neurons:
										_, occupied_links, number_of_spikes, latency_Node = \
											process_Node(
												Node_ID, Node_obj, routing_type, netlist, None, **{})

										network.nodes[Node_ID].int_packets_handled += number_of_spikes
										network.send_spike(occupied_links)
										latency_run.update(latency_Node)

									nodes_processed += 1
									if round(nodes_processed / len(network.nodes) * 1000) / 10 > ratio:
										ratio = round(nodes_processed / len(network.nodes) * 1000) / 10
										print(
											f'Running traffic analysis {routing_type} {casting_type} | {" " * (ratio < 10)}{ratio} % |',
											end='\r')

								results_traffic[topology + '_Torus' * torus_option][mapping_type][routing_type][casting_type] = \
									network.readout(settings['ignore_unused'])

								if mapping_type.lower() == 'manual' or mapping_type.lower() == 'pacman':
									network.overview_mapping_benchmark(f'{run_name} {routing_type}', mapping_type)
								elif mapping_type.lower() == 'ghost':
									network.overview_mapping_GHOST(f'{run_name} {routing_type}', mapping_type)

								results_latency[topology + '_Torus' * torus_option][mapping_type][routing_type][casting_type] = {
									'per_neuron': latency_run.copy(),
									'average': statistics.mean(latency_run.values()),
									'min': min(latency_run.values()),
									'max': max(latency_run.values()),
									'median': statistics.median(latency_run.values())}
								print("\r\033[K", end='')

								with open(run_name + '_summary.csv', 'a') as file:
									step_results = \
										results_traffic[topology + '_Torus' * torus_option][mapping_type][routing_type][casting_type]
									step_results_latency = \
										results_latency[topology + "_Torus" * torus_option][mapping_type][routing_type][casting_type]
									summary_string = \
										f'{(str(step).replace(".", ",") + ";") * bool(loop_var)}{topology + " Torus" * torus_option};' \
										f'{mapping_type};{routing_type};{casting_type};'

									data_tags = [tag for tag in step_results.keys() if not tag.startswith('(') and not tag == 'Spectrum']
									for data_tag in data_tags:
										summary_string += \
											f'{str(step_results[data_tag]["average"]).replace(".", ",")};' \
											f'{str(step_results[data_tag]["min"]).replace(".", ",")};' \
											f'{str(step_results[data_tag]["max"]).replace(".", ",")};'

									if 'Spectrum' in step_results.keys():
										summary_string += \
											f'{str(step_results["Spectrum"]).replace(".", ",")};'

									summary_string += \
										f'{str(step_results_latency["average"]).replace(".", ",")};' \
										f'{str(step_results_latency["min"]).replace(".", ",")};' \
										f'{str(step_results_latency["max"]).replace(".", ",")}\n'

									file.write(summary_string)

								sim_log.message(
									f'Completed run {routing_type} - {casting_type} in {hardware_graph.convert_time(time.time() - T1)}')

								if settings['plot_heatmaps'] and topology.lower() not in ['mesh3d', 'cube']:
									createFolder('graphs')
									os.chdir('graphs/')
									if loop_var:
										plot_heatmap(
											network.readout(settings['ignore_unused']),
											f'Link Traffic Load Heatmap - '
											f'{topology} {"Torus" * torus_option} {loop_var} {step} {mapping_type} {routing_type} {casting_type}'
										)
										plot_node_heatmap(
											network.readout(settings['ignore_unused']),
											f'Node Traffic Load Heatmap - '
											f'{topology} {"Torus" * torus_option} {loop_var} {step} {mapping_type} {routing_type} {casting_type}'
										)
									else:
										plot_heatmap(
											network.readout(settings['ignore_unused']),
											f'Link Traffic Load Heatmap - '
											f'{topology} {"Torus" * torus_option} {mapping_type} {routing_type} {casting_type}'
										)
										plot_node_heatmap(
											network.readout(settings['ignore_unused']),
											f'Node Traffic Load Heatmap - '
											f'{topology} {"Torus" * torus_option} {mapping_type} {routing_type} {casting_type}'
										)

									os.chdir('..')
							except CastingError:
								sim_log.error(
									f'An invalid combination between {topology} and {casting_type} was given. '
									f'Skip run and continue.')
								continue

					except RoutingError:
						sim_log.message(f'Routing of current run failed (see error above), abort run.\n')
						continue

				print(f'Traffic analysis completed in {hardware_graph.convert_time(time.time() - T0)}')
				sim_log.message(f'Traffic analysis completed in {hardware_graph.convert_time(time.time() - T0)}\n')

	return results_traffic, results_latency


def probability_matrix_analysis(settings, loop_var, step, run_name, i):
	if i == 0:
		# write summary file header
		with open(run_name + '_summary.csv', 'w') as file:
			if loop_var:
				file.write(f'{loop_var};')
			file.write(
				f'Topology;Mapping;Routing;Casting;'
				f'Avg spikes per router;Min spikes per router;Max spikes per router;'
				f'Avg int. spikes per router;Min int. spikes per router;Max int. spikes per router;'
				f'Avg ext. spikes per router;Min ext. spikes per router;Max ext. spikes per router;')
			if settings['secondary']:
				file.write(
					f'Average spikes per link (primary layer);Minimum spikes per link (primary layer);'
					f'Maximum spikes per link (primary layer);'
					f'Average spikes per link (secondary layer);Minimum spikes per link (secondary layer);'
					f'Maximum spikes per link (secondary layer);'
					f'Average latency;Minimum latency;Maximum latency\n')
			else:
				file.write(
					f'Average spikes per link;Minimum spikes per link;Maximum spikes per link;'
					f'Average latency;Minimum latency;Maximum latency\n')

	nr_neurons = 0
	for item in settings['connectivity_matrix']:
		nr_neurons += item['neurons']

	nr_nodes = math.ceil(nr_neurons / settings['neurons_per_node'])

	if nr_nodes == 1:
		sim_log.error(
			'Simulated NC system is a single node, and thus no communication network exists.\n'
			'Continue with next run.')
	elif nr_nodes < 1:
		sim_log.error(
			f'Simulated NC system with {nr_nodes} node(s).\n'
			'Continue with next run.')

	results_traffic = {}
	results_latency = {}

	sim_log.message(
		f'Analyse {len(settings["topologies"]) * len(settings["torus"])} different topologie(s) for given neural network.')
	i = 1
	for topology in settings['topologies']:
		for torus_option in settings['torus']:
			if (torus_option and topology.lower() in ['truenorth', 'brainscales']) or \
					(not torus_option and topology.lower() == 'spinnaker'):
				sim_log.notice(
					f'Selected combination of {topology} and {torus_option} is not possible.\n'
					f'Skip...')

			if topology.lower() == 'brainscales':
				network = hardware_graph.BrainscaleS()
			elif topology.lower().startswith('multi'):
				graph_generator, size, grid8, grid6, grid4 = graph_factory_function(topology, nr_nodes)
				network = graph_generator(size, torus_option, grid8, grid6, grid4)
			elif topology.lower().startswith('hubnetwork'):
				graph_generator, size, link_length = graph_factory_function(topology, nr_nodes)
				network = graph_generator(size, torus_option, link_length)
			else:
				graph_generator, size = graph_factory_function(topology, nr_nodes)
				if not graph_generator:
					sim_log.error(f'One of the selected topologies - {topology} - is not defined.')
					continue
				network = graph_generator(size, torus_option)

			results_traffic[topology + '_Torus' * torus_option] = {}
			results_latency[topology + '_Torus' * torus_option] = {}

			sim_log.message(f'\n{"*" * 16}\nTopology {i} of {len(settings["topologies"]) * len(settings["torus"])}')
			print(f'Topology {i} of {len(settings["topologies"]) * len(settings["torus"])}')
			i += 1
			for mapping_type in settings['mapping_types']:
				results_traffic[topology + '_Torus' * torus_option][mapping_type] = {}
				results_latency[topology + '_Torus' * torus_option][mapping_type] = {}
				network.reset_mapping()

				try:
					mapping = network.population_mapping(mapping_type)
					mapping(
						settings['neurons_per_node'], matrix=settings['connectivity_matrix'],
						matrix_file=settings['matrix_file'],
						path=settings['path'], **{})  # , **settings['mapping_args'])
					T0 = time.time()
				except MappingError:
					sim_log.message(f'Mapping failed (see error above), abort run.\n')
					continue

				for routing_type in settings['routing_types']:
					results_traffic[topology + '_Torus' * torus_option][mapping_type][routing_type] = {}
					results_latency[topology + '_Torus' * torus_option][mapping_type][routing_type] = {}
					try:
						print("\r\033[K", end='')
						for casting_type in settings['casting_types']:
							try:
								T1 = time.time()
								network.reset_traffic()
								latency_run = \
									run_simulation(network, routing_type, casting_type, matrix=settings['connectivity_matrix'], **{})

								results_traffic[topology + '_Torus' * torus_option][mapping_type][routing_type][casting_type] = \
									network.readout(settings['ignore_unused'])
								results_latency[topology + '_Torus' * torus_option][mapping_type][routing_type][casting_type] = {
									'per_neuron': latency_run.copy(),
									'average': statistics.mean(latency_run.values()),
									'min': min(latency_run.values()),
									'max': max(latency_run.values()),
									'median': statistics.median(latency_run.values())}
								print("\r\033[K", end='')

								with open(run_name + '_summary.csv', 'a') as file:
									step_results = \
										results_traffic[topology + '_Torus' * torus_option][mapping_type][routing_type][casting_type]
									step_results_latency = \
										results_latency[topology + "_Torus" * torus_option][mapping_type][routing_type][casting_type]
									summary_string = \
										f'{(str(step).replace(".", ",") + ";") * bool(loop_var)}{topology + " Torus" * torus_option};' \
										f'{mapping_type};{routing_type};{casting_type};'

									data_tags = \
										[tag for tag in step_results.keys() if not tag.startswith('(') and not tag == 'Spectrum']
									for data_tag in data_tags:
										summary_string += \
											f'{str(step_results[data_tag]["average"]).replace(".", ",")};' \
											f'{str(step_results[data_tag]["min"]).replace(".", ",")};' \
											f'{str(step_results[data_tag]["max"]).replace(".", ",")};'

									if 'Spectrum' in step_results.keys():
										summary_string += \
											f'{str(step_results["Spectrum"]["Bandwidth"]).replace(".", ",")};;;'

									summary_string += \
										f'{str(step_results_latency["average"]).replace(".", ",")};' \
										f'{str(step_results_latency["min"]).replace(".", ",")};' \
										f'{str(step_results_latency["max"]).replace(".", ",")}\n'

									file.write(summary_string)

								sim_log.message(
									f'Completed run {routing_type} - {casting_type} in {hardware_graph.convert_time(time.time() - T1)}')

								if settings['plot_heatmaps'] and topology.lower() not in ['mesh3d', 'cube']:
									createFolder('graphs')
									os.chdir('graphs/')
									if loop_var:
										plot_heatmap(
											network.readout(settings['ignore_unused']),
											f'Link Traffic Load Heatmap - '
											f'{topology} {"Torus" * torus_option} {loop_var} {step} {mapping_type} {routing_type} {casting_type}'
										)
										plot_node_heatmap(
											network.readout(settings['ignore_unused']),
											f'Node Traffic Load Heatmap - '
											f'{topology} {"Torus" * torus_option} {loop_var} {step} {mapping_type} {routing_type} {casting_type}'
										)
									else:
										plot_heatmap(
											network.readout(settings['ignore_unused']),
											f'Link Traffic Load Heatmap - '
											f'{topology} {"Torus" * torus_option} {mapping_type} {routing_type} {casting_type}'
										)
										plot_node_heatmap(
											network.readout(settings['ignore_unused']),
											f'Node Traffic Load Heatmap - '
											f'{topology} {"Torus" * torus_option} {mapping_type} {routing_type} {casting_type}'
										)

									os.chdir('..')
							except CastingError:
								sim_log.error(
									f'An invalid combination between {network.type} and {casting_type} was given. '
									f'Skip run and continue.')
								continue

					except RoutingError:
						sim_log.message(f'Routing of current run failed (see error above), abort run.\n')
						continue

				print(f'Traffic analysis completed in {hardware_graph.convert_time(time.time() - T0)}')
				sim_log.message(f'Traffic analysis completed in {hardware_graph.convert_time(time.time() - T0)}\n')

	return results_traffic, results_latency


########################################################################################################################
# Method performing the simulation of individual runs
########################################################################################################################
def run_simulation(network, routing_type, casting_type, matrix = None):
	latency = {}
	nodes_processed = 0
	ratio = 0
	processes = {}
	if casting_type.lower().endswith('flood') and network.type == 'Hub-Network':
		kargs = {'flooding_routes': network.flood_hubs()}
	else:
		kargs = {}

	print(f'Starting traffic analysis {routing_type} {casting_type}', end='\r')
	nodes_iterator = iter(network.nodes.items())
	jobs_left = len(network.nodes)
	try:
		casting_fnc = casting_factory_function(network, casting_type)
	except AttributeError:
		sim_log.error(f'Casting type "{casting_type}" not recognized')
		raise CastingError

	with concurrent.futures.ProcessPoolExecutor(max_workers=GLOBAL_VAR['nr_cores']) as executor:
		while jobs_left:
			for Node_ID, Node_obj in nodes_iterator:
				if Node_obj.neurons:
					job = executor.submit(
							casting_fnc, Node_ID, Node_obj, routing_type, None, matrix, **kargs)
					processes[job] = Node_ID
				else:
					nodes_processed += 1
					jobs_left -= 1

				if len(processes) >= GLOBAL_VAR['nr_cores']:
					break  # Maximum number of parallel processes reached, pause job submission

			for job in concurrent.futures.as_completed(processes):
				Node_ID, occupied_links, number_of_spikes, latency_Node = job.result()
				del processes[job]

				network.nodes[Node_ID].int_packets_handled += number_of_spikes
				network.send_spike(occupied_links)
				latency.update(latency_Node)

				jobs_left -= 1

				nodes_processed += 1
				if round(nodes_processed / len(network.nodes) * 1000) / 10 > ratio:
					ratio = round(nodes_processed / len(network.nodes) * 1000) / 10
					print(
						f'Running traffic analysis {routing_type} {casting_type} | {" " * (ratio < 10)}{ratio} % |',
						end='\r')

				break

	return latency


########################################################################################################################
# Automatic plot functions
########################################################################################################################
def plot_node_heatmap(data, title=None):
	nodes = {}
	for key, value in data.items():
		if key.startswith('('):
			nodes.update({string_to_tuple(key): value['int_packets_handled'] + value['ext_packets_handled']})
	x_min, x_max = \
		min(list(nodes.keys()), key=operator.itemgetter(0))[0], \
		max(list(nodes.keys()), key=operator.itemgetter(0))[0]
	y_min, y_max = \
		min(list(nodes.keys()), key=operator.itemgetter(1))[1], \
		max(list(nodes.keys()), key=operator.itemgetter(1))[1]

	grid = np.zeros(((x_max - x_min + 1), (y_max - y_min + 1)))
	for (x, y), value in nodes.items():
		grid[(y, x)] = value

	fig, ax = plt.subplots(figsize=(16, 12))
	im = ax.imshow(grid, vmin=0, vmax=max(nodes.values()), cmap=newcmp)

	# Create colorbar
	cbar = ax.figure.colorbar(im, ax=ax, aspect=40)
	cbar.ax.set_ylabel('Packets handled by node\n[spikes/timeframe]', rotation=-90, va="bottom", fontsize=16)
	cbar.ax.tick_params(labelsize=12)

	ax.set_xticks(np.arange(x_min, x_max + 1, 10))
	ax.set_yticks(np.arange(y_min, y_max + 1, 10))

	ax.tick_params(
		top=True, bottom=True, labeltop=True, labelbottom=True,
		left=True, right=True, labelleft=True, labelright=True, labelsize=12)
	ax.tick_params(axis='x', rotation=90)
	ax.set_xlim(x_min - 1/2, x_max + 1/2)
	ax.set_ylim(y_min - 1/2, y_max + 1/2)

	for edge, spine in ax.spines.items():
		spine.set_visible(False)

	if not title:
		title = 'Node Heatmap'
	ax.set_title(title, fontsize=20)

	fig.tight_layout()
	fig.savefig(f'{title}.png')
	plt.close()


def plot_heatmap(data, title=None):
	nodes = []
	edges = {}
	G = nx.DiGraph()
	for key, value in data.items():
		if key.startswith('('):
			nodes.append(string_to_tuple(key))
			for edge in value['edges'].keys():
				edges.update({(string_to_tuple(key), string_to_tuple(edge)): value['edges'][edge]['packets_handled']})

	pos = {node: node for node in nodes}

	x_min, x_max = \
		min(nodes, key=operator.itemgetter(0))[0], \
		max(nodes, key=operator.itemgetter(0))[0]
	y_min, y_max = \
		min(nodes, key=operator.itemgetter(1))[1], \
		max(nodes, key=operator.itemgetter(1))[1]

	colors = \
		[colormap_value(edges[edge], data['spikes_per_link']['max']) for edge in edges.keys()]
	G.add_nodes_from(nodes)
	G.add_edges_from(list(edges.keys()))

	# Create colorbar
	max_value = max(edges.values())
	x = max_value / 5
	stepsize = \
		int(round(x / (10 ** floor(log(x, 10))) * 2) / 2 * (10 ** floor(log(x, 10))))
	c = np.arange(0, max_value, stepsize)
	norm = matplotlib.colors.Normalize(vmin=0, vmax=max_value)
	cmap = matplotlib.cm.ScalarMappable(norm=norm, cmap=newcmp)
	cmap.set_array([])

	fig, ax = plt.subplots(figsize=(16, 12))
	nx.draw(G, pos, node_color='k', ax=ax, edge_color=colors, connectionstyle='arc3, rad = 0.1', node_size=50)
	ax.set_xlim((x_min - 1, x_max + 1))
	ax.set_ylim((y_min - 1, y_max + 1))
	cbar = fig.colorbar(cmap, ticks=c, aspect=40, shrink=((x_max - x_min)/(x_max - x_min + 2)))
	cbar.ax.set_ylabel('Nr. of packets over the link\n[spikes/timeframe]', rotation=-90, va="bottom", fontsize=16)
	cbar.ax.tick_params(labelsize=12)

	if not title:
		title = 'Node Heatmap'
	ax.set_title(title, fontsize=20)

	fig.tight_layout()
	fig.savefig(f'{title}.png')
	plt.close()


########################################################################################################################
# Set to run the main function if the file is executed
########################################################################################################################
if __name__ == '__main__':
	main()
