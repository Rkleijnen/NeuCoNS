"""
network_sim.py

Copyright (C) 2022, R. Kleijnen, Forschungszentrum Jülich,
Central Institute of Engineering, Electronics and Analytics—Electronic Systems (ZEA-2)

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
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.font_manager as font_manager
from matplotlib import cm
from math import floor, log
from matplotlib.patches import Patch

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
csfont = {'fontname':'Times New Roman', 'fontsize': 8}
font = font_manager.FontProperties(family='Times New Roman',
                                   style='normal', size=12)

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
def graph_factory_function(topology, nr_nodes, setting_args):
	kwargs = {}
	if topology.lower() == 'mesh':
		kwargs['torus'] = setting_args['torus']
		kwargs['size'] = math.ceil(math.sqrt(nr_nodes))
		if setting_args['degree'] == 4:
			return hardware_graph.Mesh4, kwargs
		if setting_args['degree'] == 6:
			return hardware_graph.Mesh6, kwargs
		if setting_args['degree'] == 8:
			return hardware_graph.Mesh8, kwargs
	elif topology.lower() == 'truenorth':
		if 'torus' in setting_args.keys():
			kwargs['torus'] = True if setting_args['torus'] == 'True' else False
		if 'chip_to_chip' in setting_args.keys():
			kwargs['chip_to_chip'] = setting_args['chip_to_chip']
		kwargs['size'] = math.ceil(math.sqrt(nr_nodes))
		return hardware_graph.TrueNorth, kwargs
	elif topology.lower() == 'brainscales':
		return hardware_graph.BrainscaleS, {}
	elif topology.lower() == 'spinnaker':
		if 'board_connectors' in setting_args.keys():
			kwargs['board_connectors'] = setting_args['board_connectors']
		kwargs['nr_boards'] = math.ceil(nr_nodes / 48)
		return hardware_graph.SpiNNaker, kwargs
	elif topology.lower().startswith('multi'):
		kwargs['size'] = math.ceil(math.sqrt(nr_nodes))
		kwargs['torus'] = setting_args['torus']
		kwargs['mesh8'] = setting_args['8Mesh']
		kwargs['mesh6'] = setting_args['6Mesh']
		kwargs['mesh4'] = setting_args['4Mesh']

		for length in kwargs['mesh8'] + kwargs['mesh6'] + kwargs['mesh4']:
			if length > (kwargs['size'] / 2):
				sim_log.notice(
					f'Higher level long range connections of length {length} '
					f'are omitted as this is larger than the largest distance between two points in the network.')

		return hardware_graph.MultiMesh, kwargs
	elif topology.lower() == 'mesh3d' or topology.lower() == 'cube':
		kwargs['torus'] = setting_args['torus']
		kwargs['size'] = int(math.ceil(nr_nodes ** (1 / 3)))
		return hardware_graph.Mesh3D, kwargs
	elif topology.lower().startswith('stacked'):
		kwargs['size'] = math.ceil(math.sqrt(nr_nodes / setting_args['layers']))
		kwargs['layers'] = setting_args['layers']
		kwargs['interconnect_top'] = setting_args['interconnect_top']
		kwargs['torus'] = setting_args['torus']
		return hardware_graph.StackedNetwork, kwargs
	else:
		return None, 0


def casting_factory_function(network, casting_type):
	casting_type = casting_type.lower()
	topology = network.type.lower()
	if (topology == 'TrueNorth' and casting_type != 'lmc') or \
		(topology in ['SpiNNaker', 'BrainscaleS'] and casting_type != 'mc'):
		raise CastingError
	elif topology.startswith('stacked'):
		if casting_type in ['pc', 'populationcast', 'population_cast']:
			return network.population_casting
		elif casting_type in ['cc', 'clustercast', 'cluster_cast']:
			return network.cluster_casting
		elif casting_type in ['mc', 'multicast', 'multi_cast']:
			return network.multicast
		else:
			raise CastingError
	elif casting_type == 'bc' or casting_type == 'broadcast':
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
	"""The top level function for the script based execution of NeuCoNS."""
	"""Do multiple lines work as well"""

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
		try:
			loop_values = [float(x) if float(x) % 1 else int(x) for x in read_config.get("Sim Settings", "Loop Values").split(',')]
		except ValueError:
			loop_values = [x.strip(' ') for x in read_config.get("Sim Settings", "Loop Values").split(',')]
	else:
		loop_var = None
		loop_values = ['single run']

	settings['routing_types'] = [item.strip() for item in read_config.get("Communication Protocol", "Routing").split(',')]
	settings['casting_types'] = [item.strip() for item in read_config.get("Communication Protocol", "Casting").split(',')]

	settings['mapping_types'] = [item.strip() for item in read_config.get("Neuron Mapping", "algorithm").split(',')]
	settings['neurons_per_node'] = int(read_config.get("Neuron Mapping", "Neurons per Node"))

	settings['mapping_args'] = {'neuron_model': read_config.get("Neuron Mapping", "neuron_model")}
	for key, value in json.loads(read_config.get("Neuron Mapping", "Mapping Parameters")).items():
		try:
			settings['mapping_args'].update({key: float(value) if float(value) % 1 else int(value)})
		except ValueError:
			if value == 'True':
				settings['mapping_args'].update({key: True})
			elif value == 'False':
				settings['mapping_args'].update({key: False})
			else:
				settings['mapping_args'].update({key: value})

	settings['topology'] = read_config.get("Network Topology", "Topology")
	settings['topology_args'] = {}
	for key, value in json.loads(read_config.get("Network Topology", "Topology Parameters")).items():
		try:
			if key in ['8Mesh', '6Mesh', '4Mesh']:
				settings['topology_args'].update({key: [int(x) for x in value.split(',')] if value else []})
			else:
				settings['topology_args'].update({key: float(value) if float(value) % 1 else int(value)})
		except ValueError:
			if value.lower() in ['true', 'false']:
				settings['topology_args'].update({key: True if value.lower() == 'true' else False})
			else:
				settings['topology_args'].update({key: value})

	if settings['topology'].lower().startswith('multi'):
		settings['secondary'] = True
	else:
		settings['secondary'] = False

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
		settings['connectivity_matrix'] = \
			netlist_generation.read_connectivity_probability_matrix('../' + settings['matrix_file'])

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
		elif loop_var in settings['topology_args'].keys():
			if step == 'True':
				settings['topology_args'].update({loop_var: True})
			elif step == 'False':
				settings['topology_args'].update({loop_var: False})
			elif step % 1:
				settings['topology_args'].update({loop_var: float(step)})
			else:
				settings['topology_args'].update({loop_var: int(step)})
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
			if settings["topology"].lower().startswith('stacked'):
				file.write(
					f'Average spikes generated per node;Minimum spikes generated per node;Maximum spikes generated per node;'
					f'Average spikes received per node;Minimum spikes received per node;Maximum spikes received per node;'
					f'Average spikes per router;Minimum spikes per router;Maximum spikes per router;'
					f'Average spikes per merger;Minimum spikes per merger;Maximum spikes per merger;')
			else:
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

	graph_generator, kwargs = graph_factory_function(settings['topology'], nr_nodes, settings['topology_args'])
	if not graph_generator:
		sim_log.fatal_error(f'The selected topologies - {settings["topology"]} - is not defined.')

	network = graph_generator(**kwargs)

	results_traffic[settings["topology"]] = {}
	results_latency[settings["topology"]] = {}

	for mapping_type in settings['mapping_types']:
		results_traffic[settings["topology"]][mapping_type] = {}
		results_latency[settings["topology"]][mapping_type] = {}
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
			results_traffic[settings["topology"]][mapping_type][routing_type] = {}
			results_latency[settings["topology"]][mapping_type][routing_type] = {}
			try:
				print("\r\033[K", end='')
				for casting_type in settings['casting_types']:
					try:
						T1 = time.time()
						network.reset_traffic()

						# Netlist based analysis does not work properly with parallelization.
						if GLOBAL_VAR['nr_cores'] > 1:
							sim_log.message(
								'The netlist based analysis does not work properly with parallelization.\n'
								'Prone to memory errors for large NNs and slower for small NNs.\n'
								)
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

						results_traffic[settings["topology"]][mapping_type][routing_type][casting_type] = \
							network.readout(settings['ignore_unused'])

						results_traffic[settings["topology"]][mapping_type][routing_type][casting_type] = \
							network.readout(settings['ignore_unused'])
						results_latency[settings["topology"]][mapping_type][routing_type][casting_type] = {
							'per_neuron': latency_run.copy(),
							'average': statistics.mean(latency_run.values()),
							'min': min(latency_run.values()),
							'max': max(latency_run.values()),
							'median': statistics.median(latency_run.values())}
						print("\r\033[K", end='')

						with open(run_name + '_summary.csv', 'a') as file:
							step_results = \
								results_traffic[settings["topology"]][mapping_type][routing_type][casting_type]
							step_results_latency = \
								results_latency[settings["topology"]][mapping_type][routing_type][casting_type]
							summary_string = \
								f'{(str(step).strip().replace(".", ",") + ";") * bool(loop_var)}{settings["topology"]}' \
								f'{" Torus" * settings["topology_args"]["torus"]};' \
								f'{mapping_type};{routing_type};{casting_type};'

							data_tags = [tag for tag in step_results.keys() if not tag.startswith('(')]
							for data_tag in data_tags:
								summary_string += \
									f'{str(step_results[data_tag]["average"]).replace(".", ",")};' \
									f'{str(step_results[data_tag]["min"]).replace(".", ",")};' \
									f'{str(step_results[data_tag]["max"]).replace(".", ",")};'

							summary_string += \
								f'{str(step_results_latency["average"]).replace(".", ",")};' \
								f'{str(step_results_latency["min"]).replace(".", ",")};' \
								f'{str(step_results_latency["max"]).replace(".", ",")}\n'

							file.write(summary_string)

						sim_log.message(
							f'Completed run {routing_type} - {casting_type} in {hardware_graph.convert_time(time.time() - T1)}')

						if settings['plot_heatmaps'] and settings["topology"].lower() not in ['mesh3d', 'cube']:
							createFolder('graphs')
							os.chdir('graphs/')
							if loop_var:
								plot_heatmap(
									network.readout(settings['ignore_unused']),
									f'Link Traffic Load Heatmap - '
									f'{settings["topology"]} {"Torus" * settings["topology_args"]["torus"]} {loop_var} {step} {mapping_type} {routing_type} {casting_type}'
								)
								plot_node_heatmap(
									network.readout(settings['ignore_unused']),
									f'Node Traffic Load Heatmap - '
									f'{settings["topology"]} {"Torus" * settings["topology_args"]["torus"]} {loop_var} {step} {mapping_type} {routing_type} {casting_type}'
								)
							else:
								plot_heatmap(
									network.readout(settings['ignore_unused']),
									f'Link Traffic Load Heatmap - '
									f'{settings["topology"]} {"Torus" * settings["topology_args"]["torus"]} {mapping_type} {routing_type} {casting_type}'
								)
								plot_node_heatmap(
									network.readout(settings['ignore_unused']),
									f'Node Traffic Load Heatmap - '
									f'{settings["topology"]} {"Torus" * settings["topology_args"]["torus"]} {mapping_type} {routing_type} {casting_type}'
								)

							os.chdir('..')
					except CastingError:
						sim_log.error(
							f'An invalid combination between {settings["topology"]} and {casting_type} was given. '
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
			if settings["topology"].lower().startswith('stacked'):
				file.write(
					f'Topology;Mapping;Routing;Casting;'
					f'Average spikes generated per node;Minimum spikes generated per node;Maximum spikes generated per node;'
					f'Average spikes received per node;Minimum spikes received per node;Maximum spikes received per node;'
					f'Average spikes per router;Minimum spikes per router;Maximum spikes per router;'
					f'Average spikes per merger;Minimum spikes per merger;Maximum spikes per merger;')
			else:
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

	nr_nodes = 0

	if settings['topology'].lower() == 'stacked' and settings['mapping_args']['neuron_model'].lower() == 'population':
		nodes = 0
		for item in settings['connectivity_matrix']:
			pop = item['population']
			if pop.endswith('23E'):
				nr_nodes += math.ceil(nodes / settings['topology_args']['layers']) * settings['topology_args']['layers']
				nodes = math.ceil(item['neurons'] / settings['neurons_per_node'])
			else:
				nodes += math.ceil(item['neurons'] / settings['neurons_per_node'])
		nr_nodes += math.ceil(nodes / settings['topology_args']['layers']) * settings['topology_args']['layers']

	elif settings['topology'].lower() == 'stacked' and settings['mapping_args']['neuron_model'].lower() == 'area':
		neurons = 0
		for item in settings['connectivity_matrix']:
			pop = item['population']
			if pop.endswith('23E'):
				nr_nodes += math.ceil(neurons / settings['neurons_per_node'] / settings['topology_args']['layers']) * \
							settings['topology_args']['layers']
				neurons = item['neurons']
			else:
				neurons += item['neurons']
		nr_nodes += math.ceil(neurons / settings['neurons_per_node'] / settings['topology_args']['layers']) * settings['topology_args']['layers']

	elif settings['topology'].lower() != 'stacked' and settings['mapping_args'][
		'neuron_model'].lower() == 'population':
		for item in settings['connectivity_matrix']:
			nr_nodes += math.ceil(item['neurons'] / settings['neurons_per_node'])
	else:
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

	graph_generator, kwargs = graph_factory_function(settings['topology'], nr_nodes, settings['topology_args'])
	if not graph_generator:
		sim_log.fatal_error(f'The selected topologies - {settings["topology"]} - is not defined.')

	network = graph_generator(**kwargs)

	results_traffic[settings["topology"]] = {}
	results_latency[settings["topology"]] = {}

	for mapping_type in settings['mapping_types']:
		results_traffic[settings["topology"]][mapping_type] = {}
		results_latency[settings["topology"]][mapping_type] = {}
		network.reset_mapping()

		try:
			mapping = network.population_mapping(mapping_type)
			mapping(
				settings['neurons_per_node'], matrix=settings['connectivity_matrix'], **settings['mapping_args'])
			T0 = time.time()
		except MappingError:
			sim_log.message(f'Mapping failed (see error above), abort run.\n')
			continue

		for routing_type in settings['routing_types']:
			results_traffic[settings["topology"]][mapping_type][routing_type] = {}
			results_latency[settings["topology"]][mapping_type][routing_type] = {}
			try:
				print("\r\033[K", end='')
				for casting_type in settings['casting_types']:
					try:
						T1 = time.time()
						network.reset_traffic()
						latency_run = \
							run_simulation(
								network, routing_type, casting_type,
								matrix=settings['connectivity_matrix'])

						results_traffic[settings["topology"]][mapping_type][routing_type][casting_type] = \
							network.readout(settings['ignore_unused'])
						results_latency[settings["topology"]][mapping_type][routing_type][casting_type] = {
							'per_neuron': latency_run.copy(),
							'average': statistics.mean(latency_run.values()),
							'min': min(latency_run.values()),
							'max': max(latency_run.values()),
							'median': statistics.median(latency_run.values())}
						print("\r\033[K", end='')

						with open(run_name + '_summary.csv', 'a') as file:
							step_results = \
								results_traffic[settings["topology"]][mapping_type][routing_type][casting_type]
							step_results_latency = \
								results_latency[settings["topology"]][mapping_type][routing_type][casting_type]
							summary_string = \
								f'{(str(step).strip().replace(".", ",") + ";") * bool(loop_var)}{settings["topology"]}' \
								f'{" Torus" * settings["topology_args"]["torus"]};' \
								f'{mapping_type};{routing_type};{casting_type};'

							data_tags = [tag for tag in step_results.keys() if not tag.startswith('(')]
							for data_tag in data_tags:
								summary_string += \
									f'{str(step_results[data_tag]["average"]).replace(".", ",")};' \
									f'{str(step_results[data_tag]["min"]).replace(".", ",")};' \
									f'{str(step_results[data_tag]["max"]).replace(".", ",")};'

							summary_string += \
								f'{str(step_results_latency["average"]).replace(".", ",")};' \
								f'{str(step_results_latency["min"]).replace(".", ",")};' \
								f'{str(step_results_latency["max"]).replace(".", ",")}\n'

							file.write(summary_string)

						sim_log.message(
							f'Completed run {routing_type} - {casting_type} in {hardware_graph.convert_time(time.time() - T1)}')

						if settings['plot_heatmaps'] and (settings["topology"].lower() not in ['mesh3d', 'cube'] and not settings["topology"].lower().startswith('stacked')):
							createFolder('graphs')
							os.chdir('graphs/')
							if loop_var:
								plot_heatmap(
									network.readout(settings['ignore_unused']),
									f'Link Traffic Load Heatmap - '
									f'{settings["topology"]} {"Torus" * settings["topology_args"]["torus"]} {loop_var} {step} {mapping_type} {routing_type} {casting_type}'
								)
								plot_node_heatmap(
									network.readout(settings['ignore_unused']),
									f'Node Traffic Load Heatmap - '
									f'{settings["topology"]} {"Torus" * settings["topology_args"]["torus"]} {loop_var} {step} {mapping_type} {routing_type} {casting_type}'
								)
							else:
								plot_heatmap(
									network.readout(settings['ignore_unused']),
									f'Link Traffic Load Heatmap - '
									f'{settings["topology"]} {"Torus" * settings["topology_args"]["torus"]} {mapping_type} {routing_type} {casting_type}'
								)
								plot_node_heatmap(
									network.readout(settings['ignore_unused']),
									f'Node Traffic Load Heatmap - '
									f'{settings["topology"]} {"Torus" * settings["topology_args"]["torus"]} {mapping_type} {routing_type} {casting_type}'
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
def run_simulation(network, routing_type, casting_type, netlist = None, matrix = None):
	"""Iterates through all nodes in the network and calculates and accumulates the communication traffic profiles of all nodes."""

	latency = {}
	nodes_processed = 0
	ratio = 0
	processes = {}
	if casting_type.lower().endswith('flood') and network.type == 'Hub-Network':
		kargs = {'flooding_routes': network.flood_hubs()}
	elif (casting_type.lower() in ['cc', 'clustercast', 'cluster_cast', 'pc', 'population_cast', 'population_cast']) \
		and network.type.startswith('Stacked'):
		kargs = {'placement_map': network.placement_dict()}
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
							casting_fnc, Node_ID, Node_obj, routing_type, netlist, matrix, **kargs)
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
# Plot functions
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
# Console functions
########################################################################################################################
def plot_mapping_multi_area(matrix_file, NpN, mapping_type, title=None):
	# The map plotting function automatically assumes a Mesh4 topology.
	# This is sufficient as the function can't handle 3D topologies
	matrix = netlist_generation.read_connectivity_probability_matrix('matrix_files/' + matrix_file)

	nr_nodes = 0
	for item in matrix:
		nr_nodes += math.ceil(item['neurons'] / NpN)

	print(math.ceil(math.sqrt(nr_nodes)))
	network = hardware_graph.Mesh4(math.ceil(math.sqrt(nr_nodes)), False)
	mapping = network.population_mapping(mapping_type)
	mapping(NpN, matrix, neuron_model='population', sort=False)
	data_grid = np.zeros(network.size)
	colors = [
		'#ffffff', '#023d6b', '#adbde3', '#006e4e', '#b9d25f', '#8b0000', '#eb5f73', '#fab45a', '#faeb5a',
		'#777777', '#CCCCCC', '#999999', '#333333', '#E0E0E0', '#888888', '#555555', '#AAAAAA', '#111111', '#040404']

	populations = ['Empty node', '23E', '23I', '4E', '4I', '5E', '5I', '6E', '6I']
	areas = []
	for population in matrix:
		if population['population'].split('-')[0] not in areas:
			areas.append(population['population'].split('-')[0])
	patches = [Patch(facecolor=color, edgecolor='k') for color in colors[:9]]
	cmap = ListedColormap(colors)

	ratio = 0
	nodes_plotted = 0

	network_size = nr_nodes
	for y in range(network.size[1]):
		for x in range(network.size[0]):
			if not network.nodes[(x, y)].neurons:
				data_grid[y][x] = 0
			elif len(network.nodes[(x, y)].neurons) > 1:
				data_grid[y][x] = 18
			else:
				area, pop = network.nodes[(x, y)].neurons[0][1].split('-')
				if area in ['FST', 'V1']:
					i = populations.index(pop)
				else:
					i = areas.index(area) % 9 + 9

				data_grid[y][x] = i
			nodes_plotted += 1

			if round(nodes_plotted / network_size * 1000) / 10 > ratio:
				ratio = round(nodes_plotted / network_size * 1000) / 10

	fig, ax = plt.subplots(figsize=(15 * cm, 8 * cm))
	ax.imshow(data_grid, cmap=cmap, vmin=0, vmax=18)
	ax.grid(which='major', axis='both', linestyle='-', color='w', linewidth=0.1)
	ax.set_xticks(np.arange(-.5, network.size[0], 1))
	ax.set_yticks(np.arange(-.5, network.size[1], 1))
	ax.set_xlim(- 1/2, network.size[0] - 1/2)
	ax.set_ylim(- 1/2, network.size[1] - 1/2)
	ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
	leg = ax.legend(patches, populations, loc='center left', bbox_to_anchor=(1.02, 0.5), prop=font)
	leg.set_title('Population mapped to node:', prop=font)

	# Add annotation to sequential multi area plotting
	# Annotate sequential mapping
	if mapping_type == 'sequential':
		ax.annotate(
			'Area "FST"', (-5, 15), (-10, 15), 'data', annotation_clip=False,
			arrowprops=dict(arrowstyle="-[, widthB=0.3, lengthB=1", facecolor='black'),
			fontproperties=font, horizontalalignment='right', verticalalignment='center')
		ax.annotate('Area "V1"', (-5, 44), (-10, 44), 'data', annotation_clip=False,
			arrowprops=dict(arrowstyle="-[, widthB=0.4, lengthB=1", facecolor='black'),
			fontproperties=font, horizontalalignment='right', verticalalignment='center')
	elif mapping_type == 'population_grouping':
		ax.annotate(
			'Area "FST"', (8, 14.5), (-10, 14.5), 'data', annotation_clip=False,
			arrowprops=dict(arrowstyle="-[, widthB=0.4, lengthB=1", facecolor='black'),
			fontproperties=font, horizontalalignment='right', verticalalignment='center')
		ax.annotate('Area "V1"', (-5, 44.5), (-10, 44.5), 'data', annotation_clip=False,
			arrowprops=dict(arrowstyle="-[, widthB=0.6, lengthB=1", facecolor='black'),
			fontproperties=font, horizontalalignment='right', verticalalignment='center')
	elif mapping_type == 'area_grouping':
		# Unsorted
		ax.annotate(
			'Area "FST"', (32, 16.5), (-10, 16.5), 'data', annotation_clip=False,
			arrowprops=dict(arrowstyle="-[, widthB=1.4, lengthB=1", facecolor='black'),
			fontproperties=font, horizontalalignment='right', verticalalignment='center')
		ax.annotate('Area "V1"', (-5, 47), (-10, 47), 'data', annotation_clip=False,
			arrowprops=dict(arrowstyle="-[, widthB=1.7, lengthB=1", facecolor='black'),
			fontproperties=font, horizontalalignment='right', verticalalignment='center')
	elif mapping_type == 'space_filling_curve':
		ax.annotate(
			'Area "FST"', (-5, 24), (-10, 24), 'data', annotation_clip=False,
			arrowprops=dict(arrowstyle="-[, widthB=1.8, lengthB=1", facecolor='black'),
			fontproperties=font, horizontalalignment='right', verticalalignment='center')
		ax.annotate('Area "V1"', (51, 53), (-10, 53), 'data', annotation_clip=False,
			arrowprops=dict(arrowstyle="-[, widthB=2.7, lengthB=1", facecolor='black'),
			fontproperties=font, horizontalalignment='right', verticalalignment='center')

	fig.subplots_adjust(left=0.2, right=0.65)
	plt.close()
	fig.savefig(title + '.png', dpi=600)


def plot_mapping_of_populations(matrix_file, NpN, mapping_type, title=None):
	# The map plotting function automatically assumes a Mesh4 topology.
	# This is sufficient as the function can't handle 3D topologies
	matrix = netlist_generation.read_connectivity_probability_matrix('matrix_files/' + matrix_file)

	nr_nodes = 0
	for item in matrix:
		nr_nodes += math.ceil(item['neurons'] / NpN)

	network = hardware_graph.Mesh4(math.ceil(math.sqrt(nr_nodes)), False)
	mapping = network.population_mapping(mapping_type)
	mapping(NpN, matrix, neuron_model='population', sort=False)
	data_grid = np.zeros(network.size)
	colors = ['#ffffff', '#023d6b', '#adbde3', '#006e4e', '#b9d25f', '#8b0000', '#eb5f73', '#fab45a', '#faeb5a',
			  '#af82b9']
	populations = ['Empty node', 'L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I', 'TC']
	patches = [Patch(facecolor=color, edgecolor='k') for color in colors]
	cmap = ListedColormap(colors)

	ratio = 0
	nodes_plotted = 0

	network_size = nr_nodes

	for y in range(network.size[1]):
		for x in range(network.size[0]):
			if not network.nodes[(x, y)].neurons:
				data_grid[y][x] = 0
			elif len(network.nodes[(x, y)].neurons) > 1:
				data_grid[y][x] = 1
			else:
				try:
					pop = network.nodes[(x, y)].neurons[0][1].split('-')[1]
				except IndexError:
					pop = network.nodes[(x, y)].neurons[0][1]
				i = populations.index(pop)
				data_grid[y][x] = i
			nodes_plotted += 1

			if round(nodes_plotted / network_size * 1000) / 10 > ratio:
				ratio = round(nodes_plotted / network_size * 1000) / 10

	fig, ax = plt.subplots(figsize=(15 * cm, 8 * cm))
	ax.imshow(data_grid, cmap=cmap)
	ax.grid(which='major', axis='both', linestyle='-', color='w', linewidth=0.3)
	ax.set_xticks(np.arange(-.5, network.size[0], 1))
	ax.set_yticks(np.arange(-.5, network.size[1], 1))
	ax.set_xlim(- 1/2, network.size[0] - 1/2)
	ax.set_ylim(- 1/2, network.size[1] - 1/2)
	ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)
	leg = ax.legend(patches, populations, loc='center left', bbox_to_anchor=(1.02, 0.5), prop=font)
	leg.set_title('Population mapped to node:', prop=font)

	fig.subplots_adjust(left=0.2, right=0.65)
	plt.close()
	fig.savefig(title + '.png', dpi=600)


########################################################################################################################
# Set to run the main function if the file is executed
########################################################################################################################
if __name__ == '__main__':
	main()
