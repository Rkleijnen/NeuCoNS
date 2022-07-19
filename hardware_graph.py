"""
hardware_graph.py

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

from math import inf, ceil, floor, exp, log2, pi, sqrt
import sim_log
import time
import random
import statistics
import ctypes

t_link = 0
t_router = 1


def probability_function(distance, pdf_type = 'Non Given', c = 0, r_half = 0, sigma = 0, lambda_ = 0, **aux_args):
	if pdf_type.lower() == 'inverse':
		return c / (r_half * distance + 1)
	elif pdf_type.lower() == 'normal' or pdf_type == 'gaussian' or pdf_type == 'bell curve':
		return c / (2 * pi) * exp(- 0.5 * (distance / sigma)**2)
	elif pdf_type.lower() == 'exponential':
		return c / (2 * pi * lambda_**2) * exp(-distance / lambda_)
	else:
		print(f'Probability density function "{pdf_type}" was not recognized')
		raise sim_log.SimError


def convert_time(time_seconds):
	seconds = int(time_seconds % 60)
	minutes = int(time_seconds / 60 % 60)
	hours = int(time_seconds / 60 / 60 % 24)
	days = int(time_seconds / 60 / 60 / 24)

	return str(days) + ' days - ' + str(hours) + ':' + str(minutes).zfill(2) + ':' + str(seconds).zfill(2)


def sign(a):
	if a >= 0:
		return 1
	else:
		return -1


def convert_path_to_route_uc(paths, destinations):
	route = {}
	for destination in destinations:
		current_route = []
		# Go backwards from the destination until the source
		current_node = destination
		try:
			while paths[current_node] != 'source':
				current_route.append((paths[current_node], current_node))
				current_node = paths[current_node]

		except KeyError:
			sim_log.error(f'The node {current_node} is not included in the path dictionary:\n{paths}')
			raise sim_log.RoutingError

		route[destination] = current_route
	return route


def convert_path_to_route(paths, destinations):
	route = {}
	visited = []
	for destination in destinations:
		current_route = []
		# Go backwards from the destination until the source
		current_node = destination
		try:
			while paths[current_node] != 'source' and current_node not in visited:
				current_route.append((paths[current_node], current_node))
				visited.append(current_node)
				current_node = paths[current_node]

		except KeyError:
			sim_log.error(f'The node {current_node} is not included in the path dictionary:\n{paths}')
			raise sim_log.RoutingError

		route[destination] = current_route

	return route


def read_populations(netlist):
	populations = {}
	try:
		for neuron in netlist:
			i = neuron.index('_')
			population_name = neuron[:i]
			if population_name not in populations:
				populations[population_name] = [neuron]
			else:
				populations[population_name].append(neuron)
	except ValueError:
		sim_log.warning(
			'No population ID found in neuron ID, netlist might not use populations.\n'
			'Return empty dictionary')
		populations = {}

	return populations


def extract_areas(matrix):
	area_dict = {}
	matrix_populations = {}
	for population in matrix:
		area, pop = population['population'].split('-')
		size = population['neurons']
		area_dict[area] = area_dict.get(area, 0) + size
		matrix_populations[area] = matrix_populations.get(area, []) + [pop]

	area_matrix = []
	for area, size in area_dict.items():
		area_matrix.append({'area': area, 'neurons': size})

	return area_matrix, matrix_populations


def sort_groups(matrix):
	sorted_list = []
	for population in matrix:
		size = population['neurons']

		i = 0
		for sorted_pop in sorted_list:
			if sorted_pop['neurons'] > size:
				i += 1
			else:
				break

		sorted_list.insert(i, population)

	return sorted_list


class Network:
	def __init__(self):
		self.nodes = {}
		self.type = 'arbitrary'

	class Node:
		def __init__(self):
			self.ext_packets_handled = 0
			self.int_packets_handled = 0
			self.neurons = []
			self.edges = {}

	class Edge:
		def __init__(self, weight = 1):
			self.weight = weight
			self.packets_handled = 0

	####################################################################################################################
	# Simulation Methods
	####################################################################################################################
	def send_spike(self, occupied_links):
		for link, FR in occupied_links.items():
			try:
				self.nodes[link[0]].edges[link[1]].packets_handled += FR
				if not link[1] == 'Spectrum':
					self.nodes[link[1]].ext_packets_handled += FR
			except KeyError:
				print(link)
				print(link[1])
				sim_log.fatal_error('')

	def reset_traffic(self):
		for node in self.nodes:
			self.nodes[node].ext_packets_handled = 0
			self.nodes[node].int_packets_handled = 0
			for edge in self.nodes[node].edges:
				self.nodes[node].edges[edge].packets_handled = 0

	def reset_mapping(self, netlist = None):
		for node in self.nodes:
			self.nodes[node].neurons = []
			self.nodes[node].ext_packets_handled = 0
			self.nodes[node].int_packets_handled = 0
			for edge in self.nodes[node].edges:
				self.nodes[node].edges[edge].packets_handled = 0

		if netlist:
			for neuron in netlist:
				try:
					del netlist[neuron]['location']
				except KeyError:
					continue

	####################################################################################################################
	# Construction Methods
	####################################################################################################################
	def add_node(self, key):
		if key in self.nodes:
			sim_log.warning(f'add_node({key}): A node with the key {key} already exists.')
		else:
			self.nodes[key] = self.Node()

	def add_edge(self, node_a, node_b, weight = 1):
		if node_a not in self.nodes:
			sim_log.error(f'add_edge({node_a}, {node_b}): Node {node_a} does not exist in the network')
		elif node_b not in self.nodes:
			sim_log.error(f'add_edge({node_a}, {node_b}): Node {node_b} does not exist in the network')
		elif node_b in self.nodes[node_a].edges:
			sim_log.warning(
				f'add_edge({node_a}, {node_b}): A link from {node_a} to {node_b} already exists.\n'
				f'Update link weight: {self.nodes[node_a].edges[node_b].weight} -> {weight}')
			self.nodes[node_a].edges[node_b].weight = weight
		else:
			self.nodes[node_a].edges[node_b] = self.Edge(weight)

	def update_edge(self, node_a, node_b, weight = 1):
		# Exactly the same as add_edge, but without the notification of the edge update
		if node_a not in self.nodes:
			sim_log.error(f'add_edge({node_a}, {node_b}): Node {node_a} does not exist in the network')
		elif node_b not in self.nodes:
			sim_log.error(f'add_edge({node_a}, {node_b}): Node {node_b} does not exist in the network')
		elif node_b in self.nodes[node_a].edges:
			self.nodes[node_a].edges[node_b].weight = weight
		else:
			sim_log.error(f'update_edge({node_a}, {node_b}): Edge ({node_a}, {node_b}) does not exist in the network')

	def remove_node(self, key):
		if key not in self.nodes:
			sim_log.warning(f'remove_node({key}): No node with the key {key} exists in the network.')
		else:
			sim_log.message(
				f'Deleting node {key} from the network.\n'
				f'Outgoing links and bidirectional incomming links are removed automatically.')
			sim_log.warning(
				f'POSSIBLE FLOATING LINKS: Uni-directional incomming links will remain and might cause problems later.')
			for outgoing_link in self.nodes[key].edges:
				self.remove_edge((key, outgoing_link), True)

			del self.nodes[key]

	def remove_edge(self, key, bi_directional = True):
		if key[0] not in self.nodes:
			sim_log.warning(f'remove_edge({key}): No node with the key {key[0]} exists in the network.')
		elif key[1] not in self.nodes[key[0]].edges:
			sim_log.warning(f'remove_edge({key}): Node {key[0]} is not connected to node {key[1]}.')
		else:
			del self.nodes[key[0]].edges[key[1]]
			if key[1] not in self.nodes:
				sim_log.message(f'Removed floating link from {key[0]} to {key[1]}.')
			else:
				sim_log.message(f'Removed link from {key[0]} to {key[1]}.')

			if bi_directional:
				sim_log.message(f'Try to removed opposite link from {key[1]} to {key[0]} as well...')
				if key[1] not in self.nodes:
					sim_log.warning(f'remove_edge({key}): No node with the key {key[1]} exists in the network.')
				elif key[0] not in self.nodes[key[1]].edges:
					sim_log.warning(
						f'remove_edge({key}, bi_directional): Could not be removed, link was uni-directional.')
				else:
					del self.nodes[key[1]].edges[key[0]]

	####################################################################################################################
	# Factory Methods
	####################################################################################################################
	def routing_algorithm(self, routing_type):
		if routing_type.lower() == 'spr' or routing_type.lower() == 'dijkstra':
			return self.dijkstra
		else:
			sim_log.error(
				f'Given routing algorithm {routing_type} is not recognized or implemented. '
				f'Continue with next iteration.')
			raise sim_log.RoutingError

	def mapping_module(self, mapping_type):
		if mapping_type.lower() == 'random':
			return self.random_placement
		elif mapping_type.lower() == 'sequentially':
			return self.sequential_placement
		else:
			sim_log.error(
				f'Given mapping algorithm {mapping_type} is not recognized or implemented. '
				f'Continue with next iteration.')
			raise sim_log.MappingError

	def population_mapping(self, mapping_type):
		if mapping_type.lower() == 'random':
			return self.random_population_mapping
		if mapping_type.lower() == 'sequential':
			return self.sequential_population_mapping
		else:
			sim_log.error(
				f'Given mapping algorithm {mapping_type} is not recognized or implemented. '
				f'Continue with next iteration.')
			raise sim_log.MappingError

	####################################################################################################################
	# Casting Methods
	####################################################################################################################
	def broadcast(self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations = list(self.nodes.keys())
		paths, distances = routing_fnc(Node_ID, destinations)
		routes = convert_path_to_route(paths, destinations)

		latency_temp = max([distances[target_Node] for target_Node in self.nodes])
		if netlist:
			number_of_spikes = sum([netlist[neuron]['FR'] for neuron in Node_object.neurons])
			latency.update({
				neuron: latency_temp for neuron in Node_object.neurons})
		elif matrix:
			populations = [item['population'] for item in matrix]
			number_of_spikes = sum(
				[tally * matrix[populations.index(population)].get('FR', 1) for (tally, population) in
				 Node_object.neurons])
			latency.update({
				population + '_' + str(i):
					latency_temp for (tally, population) in Node_object.neurons for i in range(tally)
			})
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		occupied_links = \
			{link: number_of_spikes for destination in destinations for link in routes[destination]}

		return Node_ID, occupied_links, number_of_spikes, latency

	def unicast(self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, LMC = False, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations = {}
		number_of_spikes = 0
		destinations_node = {}
		occupied_links = {}

		if netlist:
			for neuron in Node_object.neurons:
				# Convert combined connections list to weighted connections dictionary with unique entries
				if netlist[neuron]['FR'] == 0:
					continue
				destinations_neuron = [
					netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				# Remove duplicate target nodes for a single neuron if local-mc is used
				if LMC:
					destinations_neuron = list(set(destinations_neuron))

				number_of_spikes += netlist[neuron]['FR'] * len(destinations_neuron)
				destinations_node[neuron] = destinations_neuron
				for target in destinations_neuron:
					destinations[target] = destinations.get(target, 0) + netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue

				if LMC:
					probability_dict = self.destination_nodes_probabilities(population_index, matrix)

				for i in range(tally):
					if LMC:
						destinations_neuron = self.destination_nodes_matrix(probability_dict)
					else:
						destinations_neuron = self.destinations_matrix(population_index, matrix)

					neuron_ID = f'{Node_ID}_{population}_{i}'
					number_of_spikes += matrix[population_index].get('FR', 1) * len(destinations_neuron)
					destinations_node[neuron_ID] = destinations_neuron
					for target in destinations_neuron:
						destinations[target] = destinations.get(target, 0) + matrix[population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		paths, distances = routing_fnc(Node_ID, list(destinations.keys()))
		routes = convert_path_to_route_uc(paths, list(destinations.keys()))

		for destination, incomming_spikes in destinations.items():
			for link in routes[destination]:
				occupied_links[link] = occupied_links.get(link, 0) + incomming_spikes

		for neuron, destinations_neuron in destinations_node.items():
			latency.update({
				neuron: max([distances[target_Node] for target_Node in set(destinations_neuron)])
			})

		return Node_ID, occupied_links, number_of_spikes, latency

	def local_multicast(self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		return self.unicast(Node_ID, Node_object, routing_type, LMC=True, netlist=netlist, matrix=matrix, **kargs)

	def multicast(self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations_node = {}
		number_of_spikes = 0
		firing_rate = {}
		occupied_links = {}

		if netlist:
			for neuron in Node_object.neurons:
				if netlist[neuron]['FR'] == 0:
					continue
				destinations = [netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				# Remove duplicate target nodes
				destinations = list(set(destinations))
				destinations_node[neuron] = destinations
				firing_rate[neuron] = netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue

				probability_dict = self.destination_nodes_probabilities(population_index, matrix)
				for i in range(tally):
					destinations = self.destination_nodes_matrix(probability_dict)

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = destinations
					firing_rate[neuron_ID] = matrix[population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		#  If routing depends on other targets of the same neuron, routing can not be combined per node
		if routing_type.lower() == 'espr' or routing_type.lower() == 'ner':
			for neuron, destinations_neuron in destinations_node.items():
				paths, distances = routing_fnc(Node_ID, destinations_neuron)
				#TODO: Check whether this module is faster than convert_path_to_route_uc + list(set(routes))
				routes = convert_path_to_route(paths, destinations_neuron)

				number_of_spikes += firing_rate[neuron]
				for destination in destinations_neuron:
					for link in routes[destination]:
						occupied_links[link] = occupied_links.get(link, 0) + firing_rate[neuron]

				latency[neuron] = max([distances[target_Node] for target_Node in destinations_neuron])
		else:
			all_destinations = \
				[dest for destinations_neuron in destinations_node.values() for dest in destinations_neuron]

			paths, distances = routing_fnc(Node_ID, all_destinations)
			routes = convert_path_to_route_uc(paths, all_destinations)

			for neuron, destinations_neuron in destinations_node.items():
				number_of_spikes += firing_rate[neuron]
				neuron_route = []
				for destination in destinations_neuron:
					neuron_route += routes[destination]

				# Remove duplicates
				neuron_route = list(set(neuron_route))

				for link in neuron_route:
					occupied_links[link] = occupied_links.get(link, 0) + firing_rate[neuron]

				latency[neuron] = max([distances[target_Node] for target_Node in destinations_neuron])

		return Node_ID, occupied_links, number_of_spikes, latency

	####################################################################################################################
	# Routing Methods
	####################################################################################################################
	def dijkstra(self, source, destinations):
		visited = []
		# The path dictionary points to the previous node on the route coming from the source, taking the shortest path
		path = {source: 'source'}
		distance = {}

		for node in self.nodes:
			distance[node] = inf

		distance[source] = t_router

		while not all([destination in visited for destination in destinations]):
			try:
				min_distance = min([d for node, d in distance.items() if node not in visited])
				possible_nodes = [key for key, value in distance.items() if value == min_distance]
				current_node = random.choice(possible_nodes)
				if distance[current_node] == inf:
					sim_log.error(
						f'Dijkstra({source, destinations}): Not all destinations have been reached from node: {source}.\n'
						f'Check whether the following nodes are included in the graph and/or the graph is fully connected:\n'
						f'{[destination for destination in destinations if destination not in visited]}')

					raise sim_log.RoutingError()
			except ValueError:
				# No unvisited node left in distance-dict, min([]) raises ValueError.
				# Code should be unreachable while fulfilling the while condition
				sim_log.warning(
					'Dijkstra algorithm: Tried to determine the next node from an empty list.'
					'Unexpectedly reached unreachable code... Return path dictionary and continue with simulation.')
				break

			visited.append(current_node)
			if not len(visited) == len(self.nodes):  # Entire mesh has ben explored
				connected_to = [connection for connection in self.nodes[current_node].edges.keys()]
				for node in connected_to:
					distance_to_node = distance[current_node] + t_router + self.nodes[current_node].edges[
						node].weight * t_link
					if node not in visited and distance[node] > distance_to_node:
						distance[node] = distance_to_node
						path[node] = current_node

		return path, distance

	# Routing Sub-methods
	def distance_sort(self, source, destinations):
		# Create a list with destinations sorted by their approximated distance from the source.
		# The distance is approximated as |delta| (first norm).
		distances = {}
		for destination in destinations:
			distances[destination] = self.distance(source, destination)

		sorted_list = []
		while distances:
			key = min(distances, key=distances.get)
			sorted_list.append(key)
			del distances[key]

		return sorted_list

	####################################################################################################################
	# Mapping Methods
	####################################################################################################################
	def random_placement(self, NpN, netlist, **aux_args):
		sim_log.message('Determining mapping of neurons... (Randomly)')
		T0 = time.time()
		nr_neurons = len(netlist)
		ratio = 0
		neurons_placed = 0
		print(f'Mapping neurons randomly to network\t| {" " * (ratio < 10)}{ratio} % |', end='\r')
		try:
			for neuron in netlist:
				node = random.choice([N for N in self.nodes if len(self.nodes[N].neurons) < NpN])
				netlist[neuron].update({
					'location': node
				})
				self.nodes[node].neurons.append(neuron)
				neurons_placed += 1
				if round(neurons_placed / nr_neurons * 1000) / 10 > ratio:
					ratio = round(neurons_placed / nr_neurons * 1000) / 10
					print(
						f'Mapping neurons randomly to network\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

		except IndexError:
			sim_log.error(
				f'All nodes have been filled.\n{neurons_placed} neurons have been placed, '
				f'{len(netlist) - neurons_placed} have not.')
			raise sim_log.MappingError()

		sim_log.message(
			f'Mapped {len(netlist)} neurons to hardware in {convert_time(time.time() - T0)}.\n'
			f'Hardware utilization: {round(len(netlist) / (len(self.nodes) * NpN) * 1000) / 10}%\n')

	def sequential_placement(self, NpN, netlist, **aux_args):
		sim_log.message('Determining mapping of neurons... (Sequential)')
		T0 = time.time()
		nr_neurons = len(netlist)
		ratio = 0
		neurons_placed = 0
		print(f'Mapping neurons to network\t| {" " * (ratio < 10)}{ratio} % |', end='\r')
		Nodes = list(self.nodes.keys())
		try:
			i = 0
			node = Nodes[0]
			for neuron in netlist:
				if not len(self.nodes[node].neurons) < NpN:
					i += 1
					node = Nodes[i]

				netlist[neuron].update({'location': node})
				self.nodes[node].neurons.append(neuron)
				neurons_placed += 1
				if round(neurons_placed / nr_neurons * 1000) / 10 > ratio:
					ratio = round(neurons_placed / nr_neurons * 1000) / 10
					print(
						f'Mapping neurons to network\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

		except IndexError:
			sim_log.error(
				f'All nodes have been filled.\n{neurons_placed} neurons have been placed, '
				f'{len(netlist) - neurons_placed} have not.')
			raise sim_log.MappingError()

		sim_log.message(
			f'Mapped {len(netlist)} neurons to hardware in {convert_time(time.time() - T0)}.\n'
			f'Hardware utilization: {round(len(netlist) / (len(self.nodes) * NpN) * 1000) / 10}%\n')

	def sequential_population_mapping(self, NpN, matrix, **aux_args):
		sim_log.message('Determining mapping of neurons... (Population Ordered)')
		T0 = time.time()
		network_size = sum([population["neurons"] for population in matrix])
		ratio = 0
		neurons_placed = 0
		print(f'Mapping neurons to network\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

		Nodes = list(self.nodes.keys())
		current_node = Nodes.pop(0)
		mapped_to_node = 0

		for population in matrix:
			population_name = population['population']
			population_size = population['neurons']
			neuron_counter = 0

			try:
				while neuron_counter < population_size:
					if mapped_to_node < NpN:
						tally = min(population_size - neuron_counter, NpN - mapped_to_node)
						self.nodes[current_node].neurons.append((tally, population_name))
						mapped_to_node += tally
						neuron_counter += tally
						neurons_placed += tally
						if round(neurons_placed / network_size * 1000) / 10 > ratio:
							ratio = round(neurons_placed / network_size * 1000) / 10
							print(
								f'Mapping neurons randomly to network\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

					else:
						check_sum = sum([tally for tally, pop in self.nodes[current_node].neurons])
						if check_sum != NpN:
							sim_log.error(
								f'Expected number of neurons mapped to node {current_node} is {mapped_to_node}.\n'
								f'Actual number = {check_sum}.\n'
								f'Node contains the following neurons: {self.nodes[current_node].neurons}')
							raise sim_log.MappingError

						current_node = Nodes.pop(0)
						mapped_to_node = 0
			except IndexError:
				sim_log.error(
					f'All nodes have been filled.\n{neurons_placed} neurons have been placed, '
					f'{network_size - neurons_placed} have not.')
				raise sim_log.MappingError()

		sim_log.message(
			f'Mapped {network_size} neurons to hardware in {convert_time(time.time() - T0)}.\n'
			f'Hardware utilization: {round(network_size / (len(self.nodes) * NpN) * 1000) / 10}%\n')

	def random_population_mapping(self, NpN, matrix, **aux_args):
		sim_log.message('Determining mapping of neurons... (Randomly)')
		T0 = time.time()
		network_size = sum([population["neurons"] for population in matrix])
		ratio = 0
		neurons_placed = 0
		free_nodes = list(self.nodes.keys())
		print(f'Mapping neurons randomly to network\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

		for population in matrix:
			for i in range(population['neurons']):
				try:
					node = random.choice(free_nodes)
				except IndexError:
					sim_log.error(
						f'All nodes have been filled.\n{neurons_placed} neurons have been placed, '
						f'{network_size - neurons_placed} have not.')
					raise sim_log.MappingError()

				if self.nodes[node].neurons and self.nodes[node].neurons[-1][1] == population['population']:
					self.nodes[node].neurons[-1] = (
						self.nodes[node].neurons[-1][0] + 1, self.nodes[node].neurons[-1][1])
				else:
					self.nodes[node].neurons.append((1, population['population']))

				neurons_placed += 1
				if round(neurons_placed / network_size * 1000) / 10 > ratio:
					ratio = round(neurons_placed / network_size * 1000) / 10
					print(
						f'Mapping neurons randomly to network\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

				mapped_to_node = sum([tally for (tally, pop) in self.nodes[node].neurons])
				if mapped_to_node == NpN:
					del free_nodes[free_nodes.index(node)]
				elif mapped_to_node > NpN:
					sim_log.error(
						f'Node {node} has been assigned to {mapped_to_node} neurons.\n'
						f'NpN has been set to {NpN}, node is overfilled.')

		sim_log.message(
			f'Mapped {network_size} neurons to hardware in {convert_time(time.time() - T0)}.\n'
			f'Hardware utilization: {round(network_size / (len(self.nodes) * NpN) * 1000) / 10}%\n')

	####################################################################################################################
	# On-the-Fly Netlist generation (HW-dependent and Matrix defined)
	####################################################################################################################
	def fill_network(self, NpN):
		for Node in self.nodes:
			for neuron in range(NpN):
				self.nodes[Node].neurons.append(f'N{Node}_n{neuron}')

		sim_log.message(f'Filled network with {len(self.nodes.keys()) * NpN} neurons')

	def destinations_matrix(self, source_population_index, matrix):
		populations = [item['population'] for item in matrix]
		destinations = []
		for target_Node_ID, target_Node in self.nodes.items():
			for (tally, population) in target_Node.neurons:
				target_population_index = populations.index(population)
				probability = matrix[source_population_index]['connectivity_prob'][target_population_index]
				for i in range(tally):
					if random.random() <= probability:
						destinations.append(target_Node_ID)

		return destinations

	def destination_nodes_probabilities(self, source_population_index, matrix):
		populations = [item['population'] for item in matrix]
		probability_dict = {}
		for target_Node_ID, target_Node in self.nodes.items():
			probability_no_connection = 1
			for (tally, population) in target_Node.neurons:
				target_population_index = populations.index(population)
				probability = matrix[source_population_index]['connectivity_prob'][target_population_index]
				probability_no_connection = probability_no_connection * (1 - probability)**tally

			probability_dict[target_Node_ID] = 1 - probability_no_connection

		return probability_dict

	def destination_nodes_matrix(self, probability_dict):
		destinations = []
		for target_Node_ID in self.nodes.keys():
			if random.random() <= probability_dict[target_Node_ID]:
				destinations.append(target_Node_ID)

		return destinations

	# Connectivity Sub-methods
	def distance(self, node_A, node_B):
		# Calculates the distance between two nodes, The euclidean distance in this case
		deltas = (node_A[0] - node_B[0], node_A[1] - node_B[1])
		distance = (deltas[0]**2 + deltas[1]**2)**(1 / 2)
		return distance

	####################################################################################################################
	# Output Methods
	####################################################################################################################
	def readout(self, unused, compressed = None):
		output = {}
		int_lst = []
		ext_lst = []
		router_total = []
		link_lst = []
		for node in self.nodes.keys():
			output[str(node)] = {
				'int_packets_handled': self.nodes[node].int_packets_handled,
				'ext_packets_handled': self.nodes[node].ext_packets_handled,
				'edges': {}
			}
			if self.nodes[node].int_packets_handled or not unused:
				int_lst.append(self.nodes[node].int_packets_handled)
			if self.nodes[node].ext_packets_handled or not unused:
				ext_lst.append(self.nodes[node].ext_packets_handled)
			if (self.nodes[node].ext_packets_handled or self.nodes[node].int_packets_handled) or not unused:
				router_total.append(self.nodes[node].int_packets_handled + self.nodes[node].ext_packets_handled)
			for key, edge in self.nodes[node].edges.items():
				output[str(node)]['edges'][str(key)] = {'length': edge.weight, 'packets_handled': edge.packets_handled}
				if not unused or edge.packets_handled or self.nodes[key].edges[node].packets_handled:
					link_lst.append(edge.packets_handled)

		output['packets_handled_per_node'] = {
			'average': statistics.mean(router_total),
			'min': min(router_total),
			'max': max(router_total),
			'median': statistics.median(router_total)
		}
		output['int_packets_handled'] = {
			'average': statistics.mean(int_lst),
			'min': min(int_lst),
			'max': max(int_lst),
			'median': statistics.median(int_lst)
		}
		output['ext_packets_handled'] = {
			'average': statistics.mean(ext_lst),
			'min': min(ext_lst),
			'max': max(ext_lst),
			'median': statistics.median(ext_lst)
		}
		output['spikes_per_link'] = {
			'average': statistics.mean(link_lst),
			'min': min(link_lst),
			'max': max(link_lst),
			'median': statistics.median(link_lst)
		}

		if not compressed:
			return output
		elif compressed.lower() == 'node':
			return router_total
		elif compressed.lower() == 'link':
			return link_lst


class Mesh4(Network):
	def __init__(self, size, torus):
		Network.__init__(self)
		self.torus = torus
		sim_log.message('Create Mesh4 hardware graph:')
		self.size = size
		self.create_network(size)
		self.type = 'Mesh4'

	####################################################################################################################
	# Construction Methods
	####################################################################################################################
	def create_network(self, size):
		T0 = time.time()
		if type(size) == int:
			self.size = [size, size]
		else:
			self.size = size

		# Generate all nodes
		for y in range(self.size[1]):
			for x in range(self.size[0]):
				self.add_node((x, y))
		sim_log.message(f'\tGenerated {(self.size[0])} x {(self.size[1])} nodes.')

		# Create connection to all neighbours for each node
		for node in self.nodes:
			self.connect_in_grid(node)
		sim_log.message(f'\tConnected all nodes to their direct neighbours in the grid.')
		if self.torus:
			sim_log.message('\tOuter nodes are connected to opposite sides as specified by torus parameter.')
		sim_log.message(f'\tGraph generated in: {convert_time(time.time() - T0)}')

	def connect_in_grid(self, node, length = 1):
		x_coord, y_coord = node
		if x_coord < self.size[0] - length:
			self.add_edge(node, (x_coord + length, y_coord), length)
		elif self.torus:
			self.add_edge(node, ((x_coord + length) % self.size[0], y_coord), length)

		if y_coord < self.size[1] - length:
			self.add_edge(node, (x_coord, y_coord + length), length)
		elif self.torus:
			self.add_edge(node, (x_coord, (y_coord + length) % self.size[1]), length)

		if x_coord - length >= 0:
			self.add_edge(node, (x_coord - length, y_coord), length)
		elif self.torus:
			self.add_edge(node, ((x_coord - length) % self.size[0], y_coord), length)

		if y_coord - length >= 0:
			self.add_edge(node, (x_coord, y_coord - length), length)
		elif self.torus:
			self.add_edge(node, (x_coord, (y_coord - length) % self.size[1]), length)

	####################################################################################################################
	# Routing Methods
	####################################################################################################################
	def dimension_order_routing(self, source, destinations):
		# Prioritizes movement in the x-direction over movement in the y direction (and movement in the w direction)
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			deltas = self.delta(source, destination)

			x, y = 1, 0
			for delta in deltas:
				while abs(delta):
					next_node = \
						((current_node[0] + sign(delta) * x) % self.size[0],
						 (current_node[1] + sign(delta) * y) % self.size[1])
					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'dor-routing calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'dor-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= sign(delta)
				x, y = y, -1 if x == y else 1
				"""
				The statement x, y = y, -1 if x == y else 1 changes the direction of movement
					| x	| y		
				1. 	| 1 | 0		Horizontal
				2.	| 0	| 1		Vertical
				3.	| 1	| 1		Diagonal north-east	(Mesh6)
				4.	| 1	| -1	Diagonal south-east	(Mesh8)
				"""

		return path, distance

	def longest_dimension_first_routing(self, source, destinations):
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			delta_x, delta_y = self.delta(source, destination)
			deltas = {
				'x': delta_x,
				'y': delta_y
			}

			while deltas:
				temp = {key: abs(item) for key, item in deltas.items()}
				direction = max(temp, key=temp.get)
				delta = deltas[direction]

				x, y = direction == 'x', direction == 'y'

				while delta:
					next_node = \
						((current_node[0] + sign(delta) * x) % self.size[0],
						 (current_node[1] + sign(delta) * y) % self.size[1])
					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'ldfr-routing calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'ldfr-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= sign(delta)

				del deltas[direction]

		return path, distance

	def enhanced_shortest_path(self, source, destinations):
		distances = {source: t_router}
		path = {source: 'source'}
		# Sort the destinations according to their APPROXIMATED distance to the source
		destinations_sorted = self.distance_sort(source, destinations)

		while destinations_sorted:
			# Continue with the next closest unvisited destination
			current_node = destinations_sorted.pop(0)
			path.update(self.path_to_intermediate_node(source, current_node, distances))

		return path, distances

	def neighbour_exploration_routing(self, source, destinations):
		distances = {source: t_router}
		path = {source: 'source'}
		# Sort the destinations according to their APPROXIMATED distance to the source
		destinations_sorted = self.distance_sort(source, destinations)

		while destinations_sorted:
			# Continue with the next closest unvisited destination
			# Route from this destination to the nearest visited node
			current_node = destinations_sorted.pop(0)
			path.update(self.path_to_closest_node(current_node, distances))

		return path, distances

	# Routing Sub-methods
	def delta(self, source, destination):
		delta_x, delta_y = destination[0] - source[0], destination[1] - source[1]
		if self.torus:
			if abs(delta_x) > 1 / 2 * self.size[0]:
				delta_x -= sign(delta_x) * self.size[0]
			if abs(delta_y) > 1 / 2 * self.size[1]:
				delta_y -= sign(delta_y) * self.size[1]

		return [delta_x, delta_y]

	def path_to_intermediate_node(self, source, destination, distances):
		# Will refer to the nodes as follow: s=source, d=destination & c=intermediate node
		delta_sd = self.delta(source, destination)
		closest_nodes = self.distance_sort(destination, distances.keys())
		for c in closest_nodes:
			delta_sc = self.delta(source, c)
			# Check whether the current node lies on a direct path from source to destination
			if all(
					[delta_sc[i] in range(0, delta_sd[i] + sign(delta_sd[i]), sign(delta_sd[i]))
					 for i in range(len(delta_sc))]):
				path, latency = self.longest_dimension_first_routing(c, [destination])

				del path[c]
				del latency[c]
				for node, distance_to_c in latency.items():
					distances[node] = distances[c] + distance_to_c

				return path
		sim_log.fatal_error(f'No closest node found, not even the original source.')

	def path_to_closest_node(self, destination, distances):
		branch_node = self.distance_sort(destination, distances.keys())[0]
		path, latency = self.longest_dimension_first_routing(branch_node, [destination])

		del path[branch_node]
		del latency[branch_node]
		for node, distance_from_branch in latency.items():
			distances[node] = distances[branch_node] + distance_from_branch

		return path

	####################################################################################################################
	# Factory Methods
	####################################################################################################################
	def routing_algorithm(self, routing_type):
		if routing_type.lower() == 'dor':
			return self.dimension_order_routing
		elif routing_type.lower() == 'ldfr':
			return self.longest_dimension_first_routing
		elif routing_type.lower() == 'spr' or routing_type.lower() == 'dijkstra':
			return self.dijkstra
		elif routing_type.lower() == 'espr':
			return self.enhanced_shortest_path
		elif routing_type.lower() == 'ner':
			return self.neighbour_exploration_routing
		else:
			sim_log.error(
				f'Given routing algorithm {routing_type} is not recognized or implemented. '
				f'Continue with next iteration.')
			raise sim_log.RoutingError

	def population_mapping(self, mapping_type):
		if mapping_type.lower() == 'random':
			return self.random_population_mapping
		if mapping_type.lower() == 'sequential':
			return self.sequential_population_mapping
		if mapping_type.lower() == 'population_grouping':
			return self.population_grouping
		if mapping_type.lower() == 'area_grouping':
			return self.area_grouping
		else:
			sim_log.error(
				f'Given mapping algorithm {mapping_type} is not recognized or implemented. '
				f'Continue with next iteration.')
			raise sim_log.MappingError

	####################################################################################################################
	# Mapping Methods
	####################################################################################################################
	def population_grouping(self, NpN, matrix, sort = True, **aux_args):
		sim_log.message('Determining mapping of neurons... (Population Grouping)')

		T0 = time.time()
		network_size = sum([population["neurons"] for population in matrix])
		ratio = 0
		neurons_placed = 0
		print(f'Mapping neurons to network, grouped per area\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

		rows_filled = 0
		population_dict = {pop['population']: {'neurons': pop['neurons']} for pop in matrix}

		if sort:
			populations = sort_groups(matrix)

		neurons_in_row = 0
		x = 0
		dx = 1
		i = 0
		fill_path = []
		while 1:
			neurons_in_row += populations[i]['neurons']

			width = sqrt(populations[i]['neurons'] / NpN)

			while i < len(populations) - 1 and width + sqrt(populations[i + 1]['neurons'] / NpN) < self.size[0]:
				i += 1
				width += sqrt(populations[i]['neurons'] / NpN)
				neurons_in_row += populations[i]['neurons']

			nodes_required = ceil(neurons_in_row / NpN)

			Ny = ceil(nodes_required / self.size[1])
			if i == len(populations) - 1:
				Ny = self.size[1] - rows_filled

			# Generate path in which order nodes will be filled
			y = rows_filled
			dy = 1

			for nodes_reserved in range(ceil((neurons_in_row - populations[i]['neurons']) / NpN)):
				fill_path.append((x, y))
				if rows_filled <= y + dy < rows_filled + Ny:
					y += dy
				else:
					x += dx
					dy = -dy

			end_node = fill_path[-1]
			if dx > 0:
				x_lim = (fill_path[-1][0], self.size[0] - 1)
			else:
				x_lim = (0, fill_path[-1][0])

			if Ny % 2:
				x, y = fill_path[-1][0], rows_filled
			else:
				x, y = (dx + 1) / 2 * (self.size[0] - 1), rows_filled
				dx = -dx

			dy = 1

			while y < rows_filled + Ny:
				if (x, y) not in fill_path or (x, y) == end_node:
					fill_path.append((x, y))
				if x_lim[0] <= x + dx <= x_lim[1]:
					x += dx
				else:
					y += dy
					dx = -dx

			rows_filled += Ny
			neurons_in_row = - (Ny * self.size[0] * NpN - neurons_in_row)

			if i == len(populations) - 1:
				break
			else:
				i += 1

		try:
			for j in range(len(populations)):
				population_name = populations[j]['population']
				neuron_counter = 0

				while neuron_counter < populations[j]['neurons']:
					current_node = fill_path[0]
					mapped_to_node = sum([tally for tally, pop in self.nodes[current_node].neurons])
					tally = min(populations[j]['neurons'] - neuron_counter, NpN - mapped_to_node)

					self.nodes[current_node].neurons.append((tally, population_name))
					neuron_counter += tally
					neurons_placed += tally

					if round(neurons_placed / network_size * 1000) / 10 > ratio:
						ratio = round(neurons_placed / network_size * 1000) / 10
						print(
							f'Mapping neurons to network, grouped per area\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

					if mapped_to_node + tally == NpN:
						fill_path.pop(0)
					elif mapped_to_node + tally > NpN:
						raise Exception
					elif j == i - 1:
						fill_path.pop(0)
		except IndexError:
			sim_log.error(
				f'All nodes have been filled.\n{neurons_placed} neurons have been placed, '
				f'{network_size - neurons_placed} have not.')

		sim_log.message(
			f'Mapped {network_size} neurons to hardware in {convert_time(time.time() - T0)}.\n'
			f'Hardware utilization: {round(network_size / (len(self.nodes) * NpN) * 1000) / 10}%\n')

	def area_grouping(self, NpN, matrix, sort = True, **aux_args):
		sim_log.message('Determining mapping of neurons... (Area Grouping)')

		T0 = time.time()
		network_size = sum([population["neurons"] for population in matrix])
		ratio = 0
		neurons_placed = 0
		print(f'Mapping neurons to network, grouped per area\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

		rows_filled = 0
		areas, area_populations = extract_areas(matrix)
		matrix_dict = {pop['population']: {'neurons': pop['neurons'], 'connectivity_prob': pop['connectivity_prob']} for
					   pop in matrix}

		if sort:
			areas = sort_groups(areas)

		neurons_in_row = 0
		x = 0
		dx = 1
		i = 0
		fill_path = []
		while 1:
			neurons_in_row += areas[i]['neurons']

			width = sqrt(areas[i]['neurons'] / NpN)

			while i < len(areas) - 1 and width + sqrt(areas[i + 1]['neurons'] / NpN) < self.size[0]:
				# TODO: Make exceptions for final row, does it fit? just place them as squares, what to do otherwise?
				i += 1
				width += sqrt(areas[i]['neurons'] / NpN)
				neurons_in_row += areas[i]['neurons']

			nodes_required = ceil(neurons_in_row / NpN)

			Ny = ceil(nodes_required / self.size[1])
			if i == len(areas) - 1:
				Ny = self.size[1] - rows_filled

			# Generate path in which order nodes will be filled
			y = rows_filled
			dy = 1

			for nodes_reserved in range(ceil((neurons_in_row - areas[i]['neurons']) / NpN)):
				fill_path.append((x, y))
				if rows_filled <= y + dy < rows_filled + Ny:
					y += dy
				else:
					x += dx
					dy = -dy

			end_node = fill_path[-1]
			if dx > 0:
				x_lim = (fill_path[-1][0], self.size[0] - 1)
			else:
				x_lim = (0, fill_path[-1][0])

			if Ny % 2:
				x, y = fill_path[-1][0], rows_filled
			else:
				x, y = (dx + 1) / 2 * (self.size[0] - 1), rows_filled
				dx = -dx

			dy = 1

			while y < rows_filled + Ny:
				if (x, y) not in fill_path or (x, y) == end_node:
					fill_path.append((x, y))
				if x_lim[0] <= x + dx <= x_lim[1]:
					x += dx
				else:
					y += dy
					dx = -dx

			rows_filled += Ny
			neurons_in_row = - (Ny * self.size[0] * NpN - neurons_in_row)

			if i == len(areas) - 1:
				break
			else:
				i += 1

		try:
			for j in range(len(areas)):
				area_name = areas[j]['area']
				for population in area_populations[area_name]:
					population_name = area_name + '-' + population
					population_size = matrix_dict[population_name]['neurons']
					neuron_counter = 0

					while neuron_counter < population_size:
						current_node = fill_path[0]
						mapped_to_node = sum([tally for tally, pop in self.nodes[current_node].neurons])
						tally = min(population_size - neuron_counter, NpN - mapped_to_node)

						self.nodes[current_node].neurons.append((tally, population_name))
						neuron_counter += tally
						neurons_placed += tally

						if round(neurons_placed / network_size * 1000) / 10 > ratio:
							ratio = round(neurons_placed / network_size * 1000) / 10
							print(
								f'Mapping neurons to network, grouped per area\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

						if mapped_to_node + tally == NpN:
							fill_path.pop(0)
						elif mapped_to_node + tally > NpN:
							raise Exception
						elif j == i - 1:
							fill_path.pop(0)
		except IndexError:
			sim_log.error(
				f'All nodes have been filled.\n{neurons_placed} neurons have been placed, '
				f'{network_size - neurons_placed} have not.')

		sim_log.message(
			f'Mapped {network_size} neurons to hardware in {convert_time(time.time() - T0)}.\n'
			f'Hardware utilization: {round(network_size / (len(self.nodes) * NpN) * 1000) / 10}%\n')

	def distance(self, node_A, node_B):
		# Calculates the distance between two nodes, The euclidean distance in this case
		deltas = self.delta(node_A, node_B)
		distance = (deltas[0]**2 + deltas[1]**2)**(1 / 2)
		return distance


class Mesh6(Mesh4):
	def __init__(self, size, torus):
		Network.__init__(self)
		self.torus = torus
		sim_log.message('Create Mesh6 hardware graph:')
		self.create_network(size)
		self.type = 'Mesh6'

	####################################################################################################################
	# Construction Methods
	####################################################################################################################
	def connect_in_grid(self, node, length = 1):
		Mesh4.connect_in_grid(self, node, length)
		x_coord, y_coord = node
		if x_coord < self.size[0] - length and y_coord < self.size[1] - length:
			self.add_edge(node, (x_coord + length, y_coord + length), length)
		elif self.torus:
			self.add_edge(node, ((x_coord + length) % self.size[0], (y_coord + length) % self.size[1]), length)

		if x_coord - length >= 0 and y_coord - length >= 0:
			self.add_edge(node, (x_coord - length, y_coord - length), length)
		elif self.torus:
			self.add_edge(node, ((x_coord - length) % self.size[0], (y_coord - length) % self.size[1]), length)

	####################################################################################################################
	# Routing Methods
	####################################################################################################################
	def longest_dimension_first_routing(self, source, destinations):
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			delta_x, delta_y, delta_w = self.delta(source, destination)
			deltas = {
				'x': delta_x,
				'y': delta_y,
				'w': delta_w
			}

			while deltas:
				temp = {key: abs(item) for key, item in deltas.items()}
				direction = max(temp, key=temp.get)
				delta = deltas[direction]

				if direction == 'w':
					x, y = True, True
				else:
					x, y = direction == 'x', direction == 'y'

				while delta:
					next_node = \
						((current_node[0] + sign(delta) * x) % self.size[0],
						 (current_node[1] + sign(delta) * y) % self.size[1])
					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'ldfr-routing calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'ldfr-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= sign(delta)

				del deltas[direction]

		return path, distance

	# Routing Sub-methods
	def delta(self, source, destination):
		delta_x, delta_y = Mesh4.delta(self, source, destination)

		if delta_x * delta_y >= 0:  # Upper right or lower left quadrant
			delta_w = min((delta_x, delta_y), key=abs)
			return [delta_x - delta_w, delta_y - delta_w, delta_w]

		elif self.torus and \
			abs(delta_x) + abs(delta_y) > min(self.size[0] - abs(delta_x), self.size[1] - abs(delta_y)):
			if self.size[0] - abs(delta_x) < self.size[1] - abs(delta_y):
				delta_x = delta_x - sign(delta_x) * self.size[0]
			else:
				delta_y = delta_y - sign(delta_y) * self.size[1]

			delta_w = min((delta_x, delta_y), key=abs)

			return [delta_x - delta_w, delta_y - delta_w, delta_w]
		else:
			return [delta_x, delta_y, 0]

	def distance(self, node_A, node_B):
		# Calculates the distance between two nodes, The euclidean distance in this case
		deltas = Mesh4.delta(self, node_A, node_B)
		distance = (deltas[0]**2 + deltas[1]**2)**(1 / 2)
		return distance


class SpiNNaker(Mesh6):
	def __init__(self, nr_boards, board_connectors = 1):
		Network.__init__(self)
		self.torus = True
		sim_log.message('Create SpiNNaker hardware graph:')
		self.type = 'SpiNNaker'

		if nr_boards == 1:
			self.delta = self.delta_singleboard
			self.nr_boards = 1
			self.create_board()
			self.size = [8, 8]
		else:
			nr_boards = int(ceil((nr_boards / 3)**1 / 2)**2 * 3)
			self.nr_boards = nr_boards
			self.create_network(nr_boards, board_connectors)
			self.size = [(nr_boards * 48)**(1 / 2), (nr_boards * 48)**(1 / 2)]

	####################################################################################################################
	# Construction Methods
	####################################################################################################################
	def create_board(self):
		T0 = time.time()
		for y in range(8):
			for x in range(8):
				if x - y < 5 and y - x < 4:
					self.add_node((x, y))
		sim_log.message(f'\tGenerated a SpiNNaker board with 48 nodes in a hexagon grid.')

		for node in self.nodes:
			self.connect_single_board(node)
		sim_log.message('\tConnected all nodes to their direct neighbours in the grid.')
		sim_log.message('\tOuter nodes are connected to opposite sides the hexagon grid to form a torus.')
		sim_log.message(f'\tGraph generated in: {convert_time(time.time() - T0)}')

	def connect_single_board(self, node):
		x_coord, y_coord = node
		# Connect east
		if x_coord < 7 and x_coord - y_coord < 4:
			self.add_edge(node, (x_coord + 1, y_coord), 1)
		elif x_coord - y_coord == 4:
			self.add_edge(node, (x_coord - 3, y_coord + 4), 1)
		elif x_coord == 7:
			self.add_edge(node, (0, y_coord - 4), 1)

		# Connect north-east
		if x_coord < 7 and y_coord < 7:
			self.add_edge(node, (x_coord + 1, y_coord + 1), 1)
		elif y_coord == 7:
			self.add_edge(node, (x_coord - 3, 0), 1)
		else:
			self.add_edge(node, (0, y_coord - 3), 1)

		# Connect north
		if y_coord < 7 and y_coord - x_coord < 3:
			self.add_edge(node, (x_coord, y_coord + 1), 1)
		elif y_coord == 7:
			self.add_edge(node, (x_coord - 4, 0), 1)
		elif y_coord - x_coord == 3:
			self.add_edge(node, (x_coord + 4, y_coord - 3), 1)

		# Connect west
		if x_coord > 0 and y_coord - x_coord < 3:
			self.add_edge(node, (x_coord - 1, y_coord), 1)
		elif x_coord == 0:
			self.add_edge(node, (7, y_coord + 4), 1)
		elif y_coord - x_coord == 3:
			self.add_edge(node, (x_coord + 3, y_coord - 4), 1)

		# Connect south-west
		if x_coord > 0 and y_coord > 0:
			self.add_edge(node, (x_coord - 1, y_coord - 1), 1)
		elif x_coord == 0:
			self.add_edge(node, (7, y_coord + 3), 1)
		elif y_coord == 0:
			self.add_edge(node, (x_coord + 3, 7), 1)

		# Connect south
		if y_coord > 0 and x_coord - y_coord < 4:
			self.add_edge(node, (x_coord, y_coord - 1), 1)
		elif x_coord - y_coord == 4:
			self.add_edge(node, (x_coord - 4, y_coord + 3), 1)
		elif y_coord == 0:
			self.add_edge(node, (x_coord + 4, 7), 1)

	def create_network(self, nr_boards, board_connectors = 1):
		T0 = time.time()
		nr_nodes = nr_boards * 48
		size = int(nr_nodes**(1 / 2))
		self.size = [size, size]
		for y in range(size):
			for x in range(size):
				self.add_node((x, y))

		sim_log.message(f'\tGenerated a SpiNNaker network with {nr_nodes} nodes on {nr_boards} boards.')

		for node in self.nodes:
			self.connect_in_grid(node)

		for node in self.nodes:
			if not node[1] % 4 and (node[0] + node[1]) % 12 < 4:
				self.update_edge(node, (node[0], (node[1] - 1) % size), board_connectors)
				self.update_edge((node[0], (node[1] - 1) % size), node, board_connectors)

				self.update_edge(node, ((node[0] - 1) % size, (node[1] - 1) % size), board_connectors)
				self.update_edge(((node[0] - 1) % size, (node[1] - 1) % size), node, board_connectors)
			if not node[0] % 4 and (node[0] + node[1]) % 12 < 4:
				self.update_edge(node, ((node[0] - 1) % size, node[1]), board_connectors)
				self.update_edge(((node[0] - 1) % size, node[1]), node, board_connectors)

				self.update_edge(node, ((node[0] - 1) % size, (node[1] - 1) % size), board_connectors)
				self.update_edge(((node[0] - 1) % size, (node[1] - 1) % size), node, board_connectors)
			if (not (node[0] - node[1]) % 8 and 16 <= node[0] + node[1] % 24) or \
				((node[0] - node[1]) % 8 == 4 and 4 <= node[0] + node[1] % 24 < 12):
				self.update_edge(node, (node[0], (node[1] - 1) % size), board_connectors)
				self.update_edge((node[0], (node[1] - 1) % size), node, board_connectors)

				self.update_edge(node, ((node[0] - 1) % size, node[1]), board_connectors)
				self.update_edge(((node[0] - 1) % size, node[1]), node, board_connectors)

		sim_log.message('\tConnected all nodes to their direct neighbours in the grid.')
		sim_log.message('\tOuter nodes are connected to opposite sides the hexagon grid to form a torus.')
		sim_log.message(f'\tGraph generated in: {convert_time(time.time() - T0)}')

	####################################################################################################################
	# Factory Methods
	####################################################################################################################
	def routing_algorithm(self, routing_type):
		if not routing_type == 'espr':
			if routing_type == 'dor':
				sim_log.message(
					f'DOR routing is possible on the SpiNNaker system, but less effective than the ESPR routing scheme.')
				return self.dimension_order_routing
			elif routing_type == 'ldfr':
				sim_log.message(
					f'LFDR routing is possible on the SpiNNaker system, but less effective than the ESPR routing scheme.')
				return self.longest_dimension_first_routing
			elif routing_type == 'ner':
				return self.neighbour_exploration_routing
			sim_log.warning(
				f'{routing_type} unavailable. The SpiNNaker system uses the ESPR routing scheme.')
		return self.enhanced_shortest_path

	####################################################################################################################
	# Routing Methods
	####################################################################################################################
	@staticmethod
	def delta_singleboard(source, destination):
		delta_x, delta_y = destination[0] - source[0], destination[1] - source[1]

		if delta_x * delta_y >= 0:  # Upper right or lower left quadrant
			delta_w = min((delta_x, delta_y), key=abs)
			delta = [delta_x - delta_w, delta_y - delta_w, delta_w]
		else:
			delta = [delta_x, delta_y, 0]

		return delta
		if sum([abs(i) for i in delta]) <= 4:
			return delta
		else:
			if delta[2]:  # Delta W
				dx, dy = \
					delta[0] + delta[2] - sign(delta[2]) * (4 + bool(delta[0]) * 4), \
					delta[1] + delta[2] - sign(delta[2]) * (
							4 + bool(delta[1]) * 4 + bool(not (delta[0] or delta[1])) * 4)
			else:
				if delta[0] and delta[1]:
					dx = delta[0] - sign(delta[0]) * 4
					dy = delta[1] - sign(delta[1]) * 4
				else:
					if delta[0]:
						dx = delta[0] - sign(delta[0]) * 4
						dy = sign(delta[0]) * 4
					else:
						dx = sign(delta[1]) * 4
						dy = delta[1] - sign(delta[1]) * 4

			if dx * dy > 0:
				dw = min((dx, dy), key=abs)
				return [dx - dw, dy - dw, dw]
			else:
				return [dx, dy, 0]

	def dimension_order_routing(self, source, destinations):
		# Prioritizes movement in the x-direction over movement in the y direction (and movement in the w direction)
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			deltas = self.delta(source, destination)

			x, y = 1, 0
			for delta in deltas:
				while abs(delta):
					next_node = \
						((current_node[0] + sign(delta) * x),
						 (current_node[1] + sign(delta) * y))

					if next_node not in self.nodes and self.nr_boards == 1:
						x_coord, y_coord = next_node[0], next_node[1]
						if x_coord == -1:
							x_coord += 8
							y_coord += 4
						elif y_coord == 8:
							x_coord -= 4
							y_coord -= 8
						elif x_coord - y_coord == 5:
							x_coord -= 4
							y_coord += 4
						elif y_coord - x_coord == 4:
							x_coord += 4
							y_coord -= 4
						elif x_coord == 8:
							x_coord -= 8
							y_coord -= 4
						elif y_coord == -1:
							x_coord += 4
							y_coord += 8
						next_node = (x_coord, y_coord)
					else:
						next_node = (next_node[0] % self.size[0], next_node[1] % self.size[1])

					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'dor-routing calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'dor-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= sign(delta)
				x, y = y, -1 if x == y else 1
				"""
				The statement x, y = y, -1 if x == y else 1 changes the direction of movement
					| x	| y		
				1. 	| 1 | 0		Horizontal
				2.	| 0	| 1		Vertical
				3.	| 1	| 1		Diagonal north-east	(Mesh6)
				4.	| 1	| -1	Diagonal south-east	(Mesh8)
				"""

		return path, distance

	# This method is unreachable and has to be added to the routing algorithm factory method to be used.
	# Routes either diagonally -> horizontally, horizontally -> vertically or vertically -> diagonally
	def windmill_routing(self, source, destinations):
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			deltas = self.delta(source, destination)

			x, y = 1, 1
			deltas = (deltas[2], deltas[0], deltas[1])

			for delta in deltas:
				while abs(delta):
					next_node = \
						((current_node[0] + sign(delta) * x),
						 (current_node[1] + sign(delta) * y))

					if next_node not in self.nodes and self.nr_boards == 1:
						x_coord, y_coord = next_node[0], next_node[1]
						if x_coord == -1:
							x_coord += 8
							y_coord += 4
						elif y_coord == 8:
							x_coord -= 4
							y_coord -= 8
						elif x_coord - y_coord == 5:
							x_coord -= 4
							y_coord += 4
						elif y_coord - x_coord == 4:
							x_coord += 4
							y_coord -= 4
						elif x_coord == 8:
							x_coord -= 8
							y_coord -= 4
						elif y_coord == -1:
							x_coord += 4
							y_coord += 8
						next_node = (x_coord, y_coord)
					else:
						next_node = (next_node[0] % self.size[0], next_node[1] % self.size[1])

					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'dor-routing calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'dor-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= sign(delta)
				x, y = y, 0 if x == y else 1

		return path, distance

	def longest_dimension_first_routing(self, source, destinations):
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			delta_x, delta_y, delta_w = self.delta(source, destination)
			deltas = {
				'x': delta_x,
				'y': delta_y,
				'w': delta_w
			}

			while deltas:
				temp = {key: abs(item) for key, item in deltas.items()}
				direction = max(temp, key=temp.get)
				delta = deltas[direction]

				if direction == 'w':
					x, y = True, True
				else:
					x, y = direction == 'x', direction == 'y'

				while delta:
					next_node = \
						((current_node[0] + sign(delta) * x),
						 (current_node[1] + sign(delta) * y))

					if next_node not in self.nodes and self.nr_boards == 1:
						x_coord, y_coord = next_node[0], next_node[1]
						if x_coord == -1:
							x_coord += 8
							y_coord += 4
						elif y_coord == 8:
							x_coord -= 4
							y_coord -= 8
						elif x_coord - y_coord == 5:
							x_coord -= 4
							y_coord += 4
						elif y_coord - x_coord == 4:
							x_coord += 4
							y_coord -= 4
						elif x_coord == 8:
							x_coord -= 8
							y_coord -= 4
						elif y_coord == -1:
							x_coord += 4
							y_coord += 8
						next_node = (x_coord, y_coord)
					else:
						next_node = (next_node[0] % self.size[0], next_node[1] % self.size[1])

					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'ldfr-routing calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'ldfr-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= sign(delta)

				del deltas[direction]

		return path, distance


class BrainscaleS(Mesh4):
	def __init__(self):
		Network.__init__(self)
		self.torus = False
		sim_log.message('Create BrainScaleS hardware graph:')
		self.create_network()
		self.type = 'BrainscaleS'

	####################################################################################################################
	# Construction Methods
	####################################################################################################################
	def create_network(self, size = None):
		T0 = time.time()

		# Generate all nodes
		# i and j indicate coordinate of the HICANN on the wafer, each reticle contains 8 HICANN chips (Nodes)
		for y in range(2, 16):
			for x in range(4, 28):
				self.add_node((x, y))

		for y in range(4, 14):
			for x in [0, 1, 2, 3, 28, 29, 30, 31]:
				self.add_node((x, y))

		for y in [0, 1, 16, 17]:
			for x in range(12, 20):
				self.add_node((x, y))

		sim_log.message(f'\tGenerated wafer module with {len(self.nodes)} nodes.')

		# Create connection to all neighbours for each node
		for node in self.nodes:
			self.connect_in_grid(node)
		sim_log.message(f'\tConnected all nodes to their direct neighbours in the grid.\n')

		sim_log.message(f'\tGraph generated in: {convert_time(time.time() - T0)}')

	def connect_in_grid(self, node, length = None):
		x_coord, y_coord = node

		for target_node in \
			[(x_coord - 1, y_coord), (x_coord, y_coord - 1), (x_coord + 1, y_coord), (x_coord, y_coord + 1)]:
			if target_node in self.nodes:
				self.add_edge(node, target_node)

	####################################################################################################################
	# Factory Methods
	####################################################################################################################
	def routing_algorithm(self, routing_type):
		if not routing_type == 'dor':
			sim_log.warning(f'{routing_type} unavailable. The BrainScaleS system uses a routing scheme similar to DOR.')

		return self.backbone

	def backbone(self, source, destinations):
		# Backbone routing (name choosen arbitrairy) is an adaption of DOR, to facilitate the wafers round shape.
		# On the lower and upper rows of the wafer, the number of columns is lower than in the wafers center.
		# In this case, the routing is redirected upwards/downwards to a row with a larger number of columns.
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			deltas = self.delta(source, destination)

			delta = deltas[0]
			while abs(delta):
				next_node = (current_node[0] + sign(delta), current_node[1])
				if next_node not in self.nodes:
					next_node = (current_node[0], (current_node[1] + sign(deltas[1])))
					deltas[1] -= sign(deltas[1])
				elif next_node not in self.nodes[current_node].edges:
					sim_log.error(
						f'While routing from {source} to {destination}, '
						f'dor-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
					raise sim_log.RoutingError()
				else:
					delta -= sign(delta)

				path[next_node] = current_node
				distance[next_node] = \
					distance[current_node] + t_router + self.nodes[current_node].edges[next_node].weight * t_link
				current_node = next_node

			delta = deltas[1]
			while abs(delta):
				next_node = (current_node[0], current_node[1] + sign(delta))
				if next_node not in self.nodes:
					sim_log.error(
						f'While routing from {source} to {destination}, '
						f'dor-routing calculated the next node: {next_node}, but this node does not exist.')
					raise sim_log.RoutingError()
				elif next_node not in self.nodes[current_node].edges:
					sim_log.error(
						f'While routing from {source} to {destination}, '
						f'dor-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
					raise sim_log.RoutingError()

				path[next_node] = current_node
				distance[next_node] = \
					distance[current_node] + t_router + self.nodes[current_node].edges[next_node].weight * t_link
				current_node = next_node
				delta -= sign(delta)

		return path, distance


class TrueNorth(Mesh4):
	def __init__(self, size, torus = False, chip_to_chip = 1):
		Network.__init__(self)
		self.torus = torus
		sim_log.message('Create TrueNorth hardware graph:')
		self.create_network(size, chip_to_chip)
		self.type = 'TrueNorth'

	####################################################################################################################
	# Construction Methods
	####################################################################################################################
	def create_network(self, size, chip_to_chip = 1):
		T0 = time.time()
		Mesh4.create_network(self, size)

		# Replace every 32 connections at the chip boundaries with single edge objects
		sim_log.message(f'\tReplace bundles of 32 edges on chip boundaries with single edge')
		for y in range(self.size[1]):
			for x in range(1, ceil(self.size[0] / 64)):
				if y % 32 == 0:
					self.update_edge((64 * x - 1, y), (64 * x, y), chip_to_chip)
					self.update_edge((64 * x, y), (64 * x - 1, y), chip_to_chip)
				else:
					self.remove_edge(((64 * x - 1, y), (64 * x, y)), True)
					self.nodes[(64 * x - 1, y)].edges[(64 * x, y)] = \
						self.nodes[(64 * x - 1, int(y / 32) * 32)].edges[(64 * x, int(y / 32) * 32)]
					self.nodes[(64 * x, y)].edges[(64 * x - 1, y)] = \
						self.nodes[(64 * x, int(y / 32) * 32)].edges[(64 * x - 1, int(y / 32) * 32)]

		for y in range(1, ceil(self.size[1] / 64)):
			for x in range(self.size[0]):
				if x % 32 == 0:
					self.update_edge((x, 64 * y - 1), (x, 64 * y), chip_to_chip)
					self.update_edge((x, 64 * x), (x, 64 * y - 1), chip_to_chip)
				else:
					self.remove_edge(((x, 64 * y - 1), (x, 64 * y)), True)
					self.nodes[(x, 64 * y - 1)].edges[(x, 64 * y)] = \
						self.nodes[(int(x / 32) * 32, 64 * y - 1)].edges[(int(x / 32) * 32, 64 * y)]
					self.nodes[(x, 64 * y)].edges[(x, 64 * y - 1)] = \
						self.nodes[(int(x / 32) * 32, 64 * y)].edges[(int(x / 32) * 32, 64 * y - 1)]
		sim_log.message(f'\tGraph generated in: {convert_time(time.time() - T0)}')

	####################################################################################################################
	# Factory Methods
	####################################################################################################################
	def routing_algorithm(self, routing_type):
		if not routing_type == 'dor':
			sim_log.warning(f'{routing_type} unavailable. The TrueNorth system uses the DOR routing scheme.')
		return self.dimension_order_routing


class Mesh8(Mesh6):
	def __init__(self, size, torus):
		Network.__init__(self)
		self.torus = torus
		sim_log.message('Create Mesh8 hardware graph:')
		self.create_network(size)
		self.type = 'Mesh8'


	####################################################################################################################
	# Construction Methods
	####################################################################################################################
	def connect_in_grid(self, node, length = 1):
		x_coord, y_coord = node
		Mesh6.connect_in_grid(self, node, length)
		if x_coord < self.size[0] - length and y_coord - length >= 0:
			self.add_edge(node, (x_coord + length, y_coord - length), length)
		elif self.torus:
			self.add_edge(node, ((x_coord + length) % self.size[0], (y_coord - length) % self.size[1]), length)

		if x_coord - length >= 0 and y_coord < self.size[1] - length:
			self.add_edge(node, (x_coord - length, y_coord + length), length)
		elif self.torus:
			self.add_edge(node, ((x_coord - length) % self.size[0], (y_coord + length) % self.size[1]), length)


	####################################################################################################################
	# Routing Methods
	####################################################################################################################
	def longest_dimension_first_routing(self, source, destinations):
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			delta_x, delta_y, delta_w, delta_v = self.delta(source, destination)
			deltas = {
				'x': delta_x,
				'y': delta_y,
				'w': delta_w,
				'v': delta_v
			}

			while deltas:
				temp = {key: abs(item) for key, item in deltas.items()}
				direction = max(temp, key=temp.get)
				delta = deltas[direction]

				if direction == 'x':
					x, y = 1, 0
				elif direction == 'y':
					x, y = 0, 1
				elif direction == 'w':
					x, y = 1, 1
				else:
					x, y = 1, -1

				while delta:
					next_node = \
						((current_node[0] + sign(delta) * x) % self.size[0],
						 (current_node[1] + sign(delta) * y) % self.size[1])
					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'ldfr-routing calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'ldfr-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= sign(delta)

				del deltas[direction]

		return path, distance

	# Routing Sub-Methods
	def delta(self, source, destination):
		# Mesh4.delta returns relative movement (in square mesh) which already accounts for the torus shape, if set
		delta_x, delta_y = Mesh4.delta(self, source, destination)

		if delta_x * delta_y >= 0:  # Move Diagonally from bottom left to top right
			delta_w = min((delta_x, delta_y), key=abs)
			return [delta_x - delta_w, delta_y - delta_w, delta_w, 0]
		else:  # move diagonal from the top left to bottom right
			delta_v = abs(min((delta_x, delta_y), key=abs)) * sign(delta_x)
			return [delta_x - delta_v, delta_y + delta_v, 0, delta_v]


class MultiMesh(Mesh8):
	# For simplicity reasons, the routing algorithms assume max Mesh8 < 2 * min Mesh6 and max Mesh6 < 2 * min Mesh4
	def __init__(self, size, torus, mesh8 = 1, mesh6 = None, mesh4 = None):
		Network.__init__(self)
		self.torus = torus
		self.create_network(size, mesh8, mesh6, mesh4)
		try:
			self.mesh8 = mesh8 if not mesh8 == 1 else [1]
			self.mesh8.sort()
			self.mesh6 = mesh6 if mesh6 else []
			self.mesh6.sort()
			self.mesh4 = mesh4 if mesh4 else []
			self.mesh4.sort()
		except AttributeError:
			sim_log.fatal_error('Mesh sizes for the multi mesh should be given in list form')
		self.type = 'Multi-Mesh'


	####################################################################################################################
	# Construction Methods
	####################################################################################################################
	def create_network(self, size, mesh8 = None, mesh6 = None, mesh4 = None):
		sim_log.message(f'Create multi-mesh hardware graph:')
		T0 = time.time()
		if type(size) == int:
			self.size = [size, size]
		else:
			self.size = size

		# Generate all nodes
		for y in range(self.size[1]):
			for x in range(self.size[0]):
				self.add_node((x, y))
		sim_log.message(f'\tGenerated {(self.size[0])} x {(self.size[1])} nodes.')

		# Create connection to all neighbours for each node
		for node in self.nodes:
			if mesh8:
				try:
					for length in mesh8:
						Mesh8.connect_in_grid(self, node, length)
				except TypeError:
					Mesh8.connect_in_grid(self, node, mesh8)
			if mesh6:
				try:
					for length in mesh6:
						Mesh6.connect_in_grid(self, node, length)
				except TypeError:
					Mesh6.connect_in_grid(self, node, mesh6)
			if mesh4:
				try:
					for length in mesh4:
						Mesh4.connect_in_grid(self, node, length)
				except TypeError:
					Mesh4.connect_in_grid(self, node, mesh4)
		sim_log.message(f'\tConnected all nodes in the grid to a Mesh8 ({mesh8}) + Mesh6 ({mesh6}) + Mesh4 ({mesh4})')
		sim_log.message(f'\tGraph generated in: {convert_time(time.time() - T0)}')

	####################################################################################################################
	# Factory Methods
	####################################################################################################################
	# TODO: Add routing factory method

	####################################################################################################################
	# Routing Methods
	####################################################################################################################
	"""
	The following routing algorithms will not always return the shortest path to the destination,
	this depends on the combination of mesh structures.
	Each layer should at least be double the size of the previous one to prevent this.
	Direction order routing and dimension order routing differ from each other in a multi mesh.
	Dimension order routing can move in both the positive or negative direction within a dimension
	Direction order routing will only move in a single direction within a dimension
	e.g. In a Mesh4(1,5) network, while moving from (0, 0) to (4, 0),
	Dimension order routing will go (0, 0) --| +5 |--> (5, 0) --| -1 |--> (4, 0), 
		using 2 routers/hops and a combined wirelength of 6
	Direction order routing will go (0, 0) --| +1 |--> (1, 0) --| +1 |--> (2, 0) --| +1 |--> (3, 0) --| +1 |--> (4, 0)
		using 4 routers/hops and a combined wirelength of 4
	"""
	def dimension_order_routing(self, source, destinations):
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			steps = self.calculate_steps_dimensional(source, destination)
			# The list steps now indicates the amount of steps to take in each direction for each mesh size
			mesh_size = list(reversed(self.mesh8 + self.mesh6 + self.mesh4))
			x, y = 1, 0
			for direction in steps:
				if not len(mesh_size) == len(direction):
					sim_log.error('Length of mesh-size-list and list with number of hops are not equal...')
					raise sim_log.RoutingError()
				for connection_length, steps in zip(mesh_size, direction):
					for _ in range(0, steps, sign(steps)):
						next_node = (
							(current_node[0] + sign(steps) * connection_length * x) % self.size[0],
							(current_node[1] + sign(steps) * connection_length * y) % self.size[1])

						if next_node not in self.nodes:
							sim_log.error(
								f'While routing from {source} to {destination}, '
								f'dor-routing calculated the next node: {next_node}, but this node does not exist.')
							raise sim_log.RoutingError()
						elif next_node not in self.nodes[current_node].edges:
							sim_log.error(
								f'While routing from {source} to {destination}, '
								f'dor-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
							raise sim_log.RoutingError()
						else:
							path[next_node] = current_node
							distance[next_node] = \
								distance[current_node] + t_router + self.nodes[current_node].edges[
									next_node].weight * t_link
							current_node = next_node

				x, y = y, -1 if x == y else 1

		return path, distance

	def longest_dimension_first_routing(self, source, destinations):
		# The longest direction is determined again everytime when the algorithm moves down to a lower level mesh
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			steps = self.calculate_steps_dimensional(source, destination)

			# The list steps now indicates the amount of steps to take in each direction for each mesh size
			mesh_size = list(reversed(self.mesh8 + self.mesh6 + self.mesh4))
			while mesh_size:
				connection_length = mesh_size[0]
				current_step = {
					'x': steps[0].pop(0),
					'y': steps[1].pop(0),
					'w': steps[2].pop(0),
					'v': steps[3].pop(0)
				}

				while current_step:
					direction = max(current_step, key=current_step.get)
					if direction == 'x':
						x, y = 1, 0
					elif direction == 'y':
						x, y = 0, 1
					elif direction == 'w':
						x, y = 1, 1
					else:
						x, y = 1, -1

					for _ in range(0, current_step[direction], sign(current_step[direction])):
						next_node = (
							(current_node[0] + sign(current_step[direction]) * connection_length * x) % self.size[0],
							(current_node[1] + sign(current_step[direction]) * connection_length * y) % self.size[1])

						if next_node not in self.nodes:
							sim_log.error(
								f'While routing from {source} to {destination}, '
								f'ldfr-routing calculated the next node: {next_node}, but this node does not exist.')
							raise sim_log.RoutingError()
						elif next_node not in self.nodes[current_node].edges:
							sim_log.error(
								f'While routing from {source} to {destination}, '
								f'ldfr-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
							raise sim_log.RoutingError()
						else:
							path[next_node] = current_node
							distance[next_node] = \
								distance[current_node] + t_router + self.nodes[current_node].edges[
									next_node].weight * t_link
							current_node = next_node
					del current_step[direction]
				del mesh_size[0]
		return path, distance

	def direction_order_routing(self, source, destinations):
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			steps = self.calculate_steps_directional(source, destination)

			# The list steps now indicates the amount of steps to take in each direction for each mesh size
			mesh_size = list(reversed(self.mesh8 + self.mesh6 + self.mesh4))
			x, y = 1, 0
			for direction in steps:
				if not len(mesh_size) == len(direction):
					raise Exception('Thats not right, length of the two lists should be equal')
				for connection_length, steps in zip(mesh_size, direction):
					for _ in range(0, steps, sign(steps)):
						next_node = (
							(current_node[0] + sign(steps) * connection_length * x) % self.size[0],
							(current_node[1] + sign(steps) * connection_length * y) % self.size[1])

						if next_node not in self.nodes:
							sim_log.error(
								f'While routing from {source} to {destination}, '
								f'directional order-routing calculated the next node: {next_node}, but this node does not exist.')
							raise sim_log.RoutingError()
						elif next_node not in self.nodes[current_node].edges:
							sim_log.error(
								f'While routing from {source} to {destination}, '
								f'directional order-routing tried to go from {current_node} to {next_node}, '
								f'but no connection exists between these nodes.')
							raise sim_log.RoutingError()
						else:
							path[next_node] = current_node
							distance[next_node] = \
								distance[current_node] + t_router + self.nodes[current_node].edges[
									next_node].weight * t_link
							current_node = next_node

				x, y = y, -1 if x == y else 1

		return path, distance

	def longest_direction_first_routing(self, source, destinations):
		# The longest direction is determined again every time when the algorithm moves down to a lower level mesh
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			steps = self.calculate_steps_directional(source, destination)

			# The list steps now indicates the amount of steps to take in each direction for each mesh size
			mesh_size = list(reversed(self.mesh8 + self.mesh6 + self.mesh4))
			while mesh_size:
				connection_length = mesh_size[0]
				current_step = {
					'x': steps[0].pop(0),
					'y': steps[1].pop(0),
					'w': steps[2].pop(0),
					'v': steps[3].pop(0)
				}

				while current_step:
					direction = max(current_step, key=current_step.get)
					if direction == 'x':
						x, y = 1, 0
					elif direction == 'y':
						x, y = 0, 1
					elif direction == 'w':
						x, y = 1, 1
					else:
						x, y = 1, -1

					for _ in range(0, current_step[direction], sign(current_step[direction])):
						next_node = (
							(current_node[0] + sign(current_step[direction]) * connection_length * x) % self.size[0],
							(current_node[1] + sign(current_step[direction]) * connection_length * y) % self.size[1])

						if next_node not in self.nodes:
							sim_log.error(
								f'While routing from {source} to {destination}, '
								f'longest direction first routing calculated the next node: {next_node}, but this node does not exist.')
							raise sim_log.RoutingError()
						elif next_node not in self.nodes[current_node].edges:
							sim_log.error(
								f'While routing from {source} to {destination}, '
								f'longest direction first routing tried to go from {current_node} to {next_node}, '
								f'but no connection exists between these nodes.')
							raise sim_log.RoutingError()
						else:
							path[next_node] = current_node
							distance[next_node] = \
								distance[current_node] + t_router + self.nodes[current_node].edges[
									next_node].weight * t_link
							current_node = next_node
					del current_step[direction]
				del mesh_size[0]
		return path, distance

	# Routing Sub-Methods
	def calculate_steps_dimensional(self, source, destination):
		x, y = source[0], source[1]
		steps = [[], [], [], []]
		for delta_function, mesh_size in \
			[(Mesh4.delta, list(reversed(self.mesh4))),
			 (Mesh6.delta, list(reversed(self.mesh6))),
			 (Mesh8.delta, list(reversed(self.mesh8)))]:
			for connection_length in mesh_size:
				if connection_length:
					deltas = list(delta_function(self, (x, y), destination))
					while not len(deltas) == 4:
						deltas.append(0)
					previous_deltas = deltas.copy()
					for i in range(len(deltas)):
						steps[i].append(0)
						while abs(deltas[i]) > abs(abs(deltas[i]) - connection_length):
							if abs(deltas[i]) < connection_length and not self.torus:
								# Next step goes past the destination before turning around, check whether you still remain within the grid
								temp_deltas = deltas.copy()
								temp_deltas[i] -= sign(deltas[i]) * connection_length
								next_position = (
									destination[0] - temp_deltas[0] - temp_deltas[2] - temp_deltas[3],
									destination[1] - temp_deltas[1] - temp_deltas[2] + temp_deltas[3])
								if not (0 <= next_position[0] < self.size[0] and 0 <= next_position[1] < self.size[1]):
									# Next step in this layer would jump outside the bounds of the network
									break
							steps[i][-1] += sign(deltas[i])
							deltas[i] -= sign(deltas[i]) * connection_length

						i += 1
					delta_travelled = \
						[(previous_delta - delta) for delta, previous_delta in zip(deltas, previous_deltas)]
					x += delta_travelled[0] + delta_travelled[2] + delta_travelled[3]
					y += delta_travelled[1] + delta_travelled[2] - delta_travelled[3]
		return steps

	def calculate_steps_directional(self, source, destination):
		x, y = source[0], source[1]
		steps = [[], [], [], []]
		for delta_function, mesh_size in \
			[(Mesh4.delta, reversed(self.mesh4)),
			 (Mesh6.delta, reversed(self.mesh6)),
			 (Mesh8.delta, reversed(self.mesh8))]:

			for connection_length in mesh_size:
				if connection_length:
					deltas = list(delta_function(self, (x, y), destination))
					while not len(deltas) == 4:
						deltas.append(0)
					previous_deltas = deltas.copy()
					for i in range(len(deltas)):
						steps[i].append(0)
						while abs(deltas[i]) >= connection_length:
							steps[i][-1] += sign(deltas[i])
							deltas[i] -= sign(deltas[i]) * connection_length
						i += 1
					delta_travelled = \
						[(previous_delta - delta) for delta, previous_delta in zip(deltas, previous_deltas)]
					x += delta_travelled[0] + delta_travelled[2] + delta_travelled[3]
					y += delta_travelled[1] + delta_travelled[2] - delta_travelled[3]
		return steps

	def routing_algorithm(self, routing_type):
		if routing_type.lower() == 'dor':
			return self.dimension_order_routing
		elif routing_type.lower() == 'ldfr':
			return self.longest_dimension_first_routing
		elif routing_type.lower() == 'directionor' or routing_type.lower() == 'dmor':
			return self.direction_order_routing
		elif routing_type.lower() == 'ldirectionfr' or routing_type.lower() == 'ldm':
			return self.longest_direction_first_routing
		elif routing_type.lower() == 'spr' or routing_type.lower() == 'dijkstra':
			return self.dijkstra
		else:
			sim_log.error(
				f'Given routing algorithm {routing_type} is not recognized or implemented. '
				f'Continue with next iteration.')
			raise sim_log.RoutingError

	"""
	================
	Casting methods
	================
	"""
	def broadcastfirst_unicast(self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations = {}
		number_of_spikes = 0
		destinations_node = {}

		routes_bc, distances_to_hubs, broadcast_destinations = self.broadcastfirst_prephase(Node_ID, routing_fnc)

		# Determine destinations
		if netlist:
			for neuron in Node_object.neurons:
				# Convert combined connections list to weighted connections dictionary with unique entries
				if netlist[neuron]['FR'] == 0:
					continue
				destinations_neuron = \
					[netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				destinations_node[neuron] = destinations_neuron
				number_of_spikes += netlist[neuron]['FR']
				for target in destinations_neuron:
					destinations[target] = destinations.get(target, 0) + netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue

				for i in range(tally):
					destinations_neuron = self.destinations_matrix(population_index, matrix)

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = destinations_neuron
					number_of_spikes += matrix[population_index].get('FR', 1)
					for target in destinations_neuron:
						destinations[target] = destinations.get(target, 0) + matrix[population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		occupied_links, routes_phase3, distances = \
			self.broadcastfirst_postphase(
				broadcast_destinations, routes_bc, number_of_spikes, destinations_node, Node_ID, distances_to_hubs, routing_fnc)

		for destination, firerate in destinations.items():
			for link in routes_phase3[destination]:
				occupied_links[link] = occupied_links.get(link, 0) + firerate

		for neuron, destinations_neuron in destinations_node.items():
			latency.update({
				neuron: max([distances[target_Node] for target_Node in set(destinations_neuron)])
			})

		return Node_ID, occupied_links, number_of_spikes, latency

	def broadcastfirst_local_multicast(
		self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations = {}
		number_of_spikes = 0
		destinations_node = {}

		routes_pre_phase3, distances_to_hubs = self.broadcastfirst_prephase(Node_ID, routing_fnc)

		# Determine destinations
		if netlist:
			for neuron in Node_object.neurons:
				# Convert combined connections list to weighted connections dictionary with unique entries
				if netlist[neuron]['FR'] == 0:
					continue
				destinations_neuron = \
					[netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				# Remove duplicate target nodes for a single neuron
				destinations_neuron = list(set(destinations_neuron))

				destinations_node[neuron] = destinations_neuron
				number_of_spikes += netlist[neuron]['FR']
				for target in destinations_neuron:
					destinations[target] = destinations.get(target, 0) + netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue

				for i in range(tally):
					destinations_neuron = self.destinations_nodes_matrix(population_index, matrix)

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = destinations_neuron
					number_of_spikes += matrix[population_index].get('FR', 1)
					for target in destinations_neuron:
						destinations[target] = destinations.get(target, 0) + matrix[
							population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		occupied_links, routes_phase3, distances = \
			self.broadcastfirst_postphase(
				destinations_node, routes_pre_phase3, number_of_spikes, distances_to_hubs, routing_fnc)

		for destination, firerate in destinations.items():
			for link in routes_phase3[destination]:
				occupied_links[link] = occupied_links.get(link, 0) + firerate

		for neuron, destinations_neuron in destinations_node.items():
			latency.update({
				neuron: max([distances[target_Node] for target_Node in set(destinations_neuron)])
			})

		return Node_ID, occupied_links, number_of_spikes, latency

	def broadcastfirst_multicast(self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations_node = {}
		firing_rate = {}

		routes_pre_phase3, distances_to_hubs = self.broadcastfirst_prephase(Node_ID, routing_fnc)

		# Determine destinations
		if netlist:
			for neuron in Node_object.neurons:
				if netlist[neuron]['FR'] == 0:
					continue
				destinations = \
					[netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				# Remove duplicate target nodes
				destinations = list(set(destinations))
				destinations_node[neuron] = destinations
				firing_rate[neuron] = netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue
				for i in range(tally):
					destinations = self.destinations_nodes_matrix(population_index, matrix)

					# Remove duplicate target nodes
					destinations = list(set(destinations))

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = destinations
					firing_rate[neuron_ID] = matrix[population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		number_of_spikes = sum(firing_rate.values())

		occupied_links, routes_phase3, distances = \
			self.broadcastfirst_postphase(
				destinations_node, routes_pre_phase3, number_of_spikes, distances_to_hubs,  routing_fnc)

		for neuron, destinations_neuron in destinations_node.items():
			neuron_route = []
			for destination in destinations_neuron:
				neuron_route += routes_phase3[destination]

			# Remove duplicates
			neuron_route = list(set(neuron_route))

			for link in neuron_route:
				occupied_links[link] = occupied_links.get(link, 0) + firing_rate[neuron]

			latency[neuron] = max([distances[target_Node] for target_Node in destinations_neuron])

		return Node_ID, occupied_links, number_of_spikes, latency

	def flood(
		self, Node_ID, Node_object, routing_type, casting, netlist = None, matrix = None, flooding_routes = None):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations_node = {}  # only store the hubs which cover at least one destination node
		number_of_spikes = 0
		firing_rate = {}
		occupied_links = {}

		# Phase 0: Determine all hubs that cover at least one destination node
		if netlist:
			for neuron in Node_object.neurons:
				# Convert combined connections list to weighted connections dictionary with unique entries
				if netlist[neuron]['FR'] == 0:
					continue
				destinations = [netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				# Remove duplicate target nodes
				destinations = list(set(destinations))
				destinations_node[neuron] = list(set([self.closest_hub(Node_ID, destination) for destination in destinations]))
				firing_rate[neuron] = netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue
				for i in range(tally):
					destinations = self.destinations_nodes_matrix(population_index, matrix)

					# Remove duplicate target nodes
					destinations = list(set(destinations))

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = list(
						set([self.closest_hub(Node_ID, destination) for destination in destinations]))
					firing_rate[neuron_ID] = matrix[population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		destination_hubs = list(set([dest for dest_list in destinations_node.values() for dest in dest_list]))

		print(destination_hubs)
		# Phase 1: Route to the hubs of interest
		paths, distances = routing_fnc(Node_ID, destination_hubs)

		for neuron, destinations_neuron in destinations_node.items():
			routes_to_hubs = []
			flooding = []
			for destination in destinations_neuron:
				current_node = destination
				while paths[current_node] != 'source':
					routes_to_hubs.append((paths[current_node], current_node))
					current_node = paths[current_node]

				# Phase 3: Flood/Broadcast corresponding hubs
				flooding += flooding_routes[0][destination]

			if casting == 'mc':
				routes_to_hubs = list(set(routes_to_hubs))
				number_of_spikes += firing_rate[neuron]
			elif casting == 'uc':
				number_of_spikes += firing_rate[neuron]
			elif casting == 'full-uc':
				number_of_spikes += firing_rate[neuron] * len(destinations_neuron)

			for link in routes_to_hubs + flooding:
				occupied_links[link] = occupied_links.get(link, 0) + firing_rate[neuron]

			latency[neuron] = max(
				[distances[destination] + flooding_routes[1][destination] for destination in destinations_neuron])

		return Node_ID, occupied_links, number_of_spikes, latency

	def uc_flood(
		self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, flooding_routes = None):
		return self.flood(Node_ID, Node_object, routing_type, 'uc', netlist, matrix, flooding_routes)

	def mc_flood(
		self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, flooding_routes = None):
		return self.flood(Node_ID, Node_object, routing_type, 'mc', netlist, matrix, flooding_routes)

	def full_uc_flood(
		self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, flooding_routes = None):
		return self.flood(Node_ID, Node_object, routing_type, 'full-uc', netlist, matrix, flooding_routes)

	# Casting Sub-Methods
	def broadcastfirst_prephase(self, Node_ID, routing_fnc):
		longest_link = max(self.mesh8 + self.mesh6 + self.mesh4)
		x, y = Node_ID
		broadcast_destinations = [
			(xi, yi) for xi in range(self.size) for yi in range(self.size)
			if xi % longest_link == x and yi % longest_link == y
		]

		paths_bc, distances = routing_fnc(Node_ID, broadcast_destinations)
		routes_bc = [(previous_node, node) for node, previous_node in paths_bc.items() if previous_node != 'source']

		return routes_bc, distances, broadcast_destinations

	def broadcastfirst_postphase(
		self, broadcast_destinations, routes_bc, number_of_spikes, destinations_node, source_ID, distances_to_hubs,
		routing_fnc):
		hubs = {hub: [] for hub in broadcast_destinations}
		occupied_links = {}
		for link in routes_bc:
			occupied_links[link] = number_of_spikes

		for target_ID in destinations_node:
			hubs[self.closest_hub(source_ID, target_ID)].append(target_ID)

		# Phase 3 of casting: From the hubs to all surround destination nodes
		routes_phase3, distances = self.route_hubs_to_nodes(broadcast_destinations, distances_to_hubs, routing_fnc)

		return occupied_links, routes_phase3, distances

	def closest_hub(self, source_ID, target_ID):
		longest_link = max(self.mesh8 + self.mesh6 + self.mesh4)

		x0, y0 = source_ID
		x1, y1 = target_ID

		dx = (x1 - x0) % longest_link
		dy = (y1 - y0) % longest_link

		if dx > longest_link / 2:
			dx -= longest_link
		if dy > longest_link / 2:
			dy -= longest_link

		return x1 - dx, y1 - dy

	@staticmethod
	def route_hubs_to_nodes(hubs, distances_to_hubs, routing_fnc):
		distance = distances_to_hubs
		routes = {}
		for hub, local_destinations in hubs.items():
			path_from_hub, distance_from_hub = routing_fnc(hub, local_destinations)

			for destination in local_destinations:
				routes[destination] = []
				current_node = destination
				while current_node != hub:
					previous_node = path_from_hub[current_node]
					routes[destination].append((previous_node, current_node))
					current_node = previous_node

				distance[destination] = distance[hub] + distance_from_hub[destination] - t_router
			return routes, distance

	"""
	================
	Output functions
	================
	"""
	def readout(self, unused, compressed = None):
		output = {}
		int_lst = []
		ext_lst = []
		router_total = []
		link_lst1 = []
		link_lst2 = []
		for node in self.nodes.keys():
			output[str(node)] = {
				'int_packets_handled': self.nodes[node].int_packets_handled,
				'ext_packets_handled': self.nodes[node].ext_packets_handled,
				'edges': {}
			}
			if self.nodes[node].int_packets_handled or not unused:
				int_lst.append(self.nodes[node].int_packets_handled)
			if self.nodes[node].ext_packets_handled or not unused:
				ext_lst.append(self.nodes[node].ext_packets_handled)
			if (self.nodes[node].ext_packets_handled or self.nodes[node].int_packets_handled) or not unused:
				router_total.append(self.nodes[node].int_packets_handled + self.nodes[node].ext_packets_handled)
			for key, edge in self.nodes[node].edges.items():
				output[str(node)]['edges'][str(key)] = {'length': edge.weight, 'packets_handled': edge.packets_handled}
				if not unused or edge.packets_handled or self.nodes[key].edges[node].packets_handled:
					if edge.weight == 1:
						link_lst1.append(edge.packets_handled)
					else:
						link_lst2.append(edge.packets_handled)

		output['int_packets_handled'] = {
			'average': statistics.mean(int_lst),
			'min': min(int_lst),
			'max': max(int_lst),
			'median': statistics.median(int_lst)
		}
		output['ext_packets_handled'] = {
			'average': statistics.mean(ext_lst),
			'min': min(ext_lst),
			'max': max(ext_lst),
			'median': statistics.median(ext_lst)
		}
		output['packets_handled_per_node'] = {
			'average': statistics.mean(router_total),
			'min': min(router_total),
			'max': max(router_total),
			'median': statistics.median(router_total)
		}
		output['spikes_per_link_primary_layer'] = {
			'average': statistics.mean(link_lst1),
			'min': min(link_lst1),
			'max': max(link_lst1),
			'median': statistics.median(link_lst1)
		}
		output['spikes_per_link_secondary_layer'] = {
			'average': statistics.mean(link_lst2),
			'min': min(link_lst2),
			'max': max(link_lst2),
			'median': statistics.median(link_lst2)
		}

		if not compressed:
			return output
		elif compressed.lower() == 'node':
			return router_total
		elif compressed.lower() == 'link':
			return link_lst1


class Mesh3D(Mesh4):
	def __init__(self, size, torus):
		Network.__init__(self)
		self.torus = torus
		self.size = size
		self.create_network(size)
		self.type = 'Mesh3D'

	####################################################################################################################
	# Construction Methods
	####################################################################################################################
	def create_network(self, size):
		T0 = time.time()
		if type(size) == int:
			self.size = [size, size, size]
		else:
			self.size = size

		# Generate all nodes
		for z in range(self.size[2]):
			for y in range(self.size[1]):
				for x in range(self.size[0]):
					self.add_node((x, y, z))
		sim_log.message(f'Generated {self.size} x {self.size} x {self.size} nodes.')

		# Create connection to all neighbours for each node
		for node in self.nodes:
			self.connect_in_grid(node)
		sim_log.message(f'\tConnected all nodes to their direct neighbours in the grid.')
		if self.torus:
			sim_log.message('\tOuter nodes are connected to opposite sides as specified by torus parameter.')
		sim_log.message(f'\tGraph generated in: {convert_time(time.time() - T0)}')

	def connect_in_grid(self, node, length = 1):
		x_coord, y_coord, z_coord = node
		if x_coord < self.size[0] - length:
			self.add_edge(node, (x_coord + length, y_coord, z_coord), length)
		elif self.torus:
			self.add_edge(node, ((x_coord + length) % self.size[0], y_coord, z_coord), length)

		if y_coord < self.size[1] - length:
			self.add_edge(node, (x_coord, y_coord + length, z_coord), length)
		elif self.torus:
			self.add_edge(node, (x_coord, (y_coord + length) % self.size[1], z_coord), length)

		if z_coord < self.size[2] - length:
			self.add_edge(node, (x_coord, y_coord, z_coord + length), length)
		elif self.torus:
			self.add_edge(node, (x_coord, y_coord, (z_coord + length) % self.size[2]), length)

		if x_coord - length >= 0:
			self.add_edge(node, (x_coord - length, y_coord, z_coord), length)
		elif self.torus:
			self.add_edge(node, ((x_coord - length) % self.size[0], y_coord, z_coord), length)

		if y_coord - length >= 0:
			self.add_edge(node, (x_coord, y_coord - length, z_coord), length)
		elif self.torus:
			self.add_edge(node, (x_coord, (y_coord - length) % self.size[1], z_coord), length)

		if z_coord - length >= 0:
			self.add_edge(node, (x_coord, y_coord, z_coord - length), length)
		elif self.torus:
			self.add_edge(node, (x_coord, y_coord, (z_coord - length) % self.size[2]), length)


	####################################################################################################################
	# Factory Methods
	####################################################################################################################
	def population_mapping(self, mapping_type):
		if mapping_type.lower() == 'random':
			return self.random_population_mapping
		if mapping_type.lower() == 'sequential':
			return self.sequential_population_mapping
		else:
			sim_log.error(
				f'Given mapping algorithm {mapping_type} is not recognized or implemented. '
				f'Continue with next iteration.')
			raise sim_log.MappingError

	####################################################################################################################
	# Routing Methods
	####################################################################################################################
	def dimension_order_routing(self, source, destinations):
		# Prioritizes movement in the x-direction over movement in the y-direction and z-direction
		# When long range connections are used, this algorithm might not always return the shortest path
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			deltas = self.delta(source, destination)
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			elif (destination[0], destination[1], source[2]) in path:
				# Path to the destinations row has already been visited prior, continue in z-direction from there
				current_node = (destination[0], destination[1], source[2])
				deltas[0] = 0
				deltas[1] = 0
			elif (destination[0], source[1], source[2]) in path:
				# Path to the destinations column has already been visited prior, continue in y-direction from there
				current_node = (destination[0], source[1], source[2])
				deltas[0] = 0
			else:
				current_node = source

			x, y, z = 1, 0, 0
			for delta in deltas:
				while abs(delta):
					next_node = \
						((current_node[0] + sign(delta) * x) % self.size[0],
						 (current_node[1] + sign(delta) * y) % self.size[1],
						 (current_node[2] + sign(delta) * z) % self.size[2])
					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'dor-routing calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'dor-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= sign(delta)
				x, y, z = 0, x, y
				"""
				The statement x, y = y, -1 if x == y else 1 changes the direction of movement
					| x	| y	| z	
				1. 	| 1 | 0	| 0	Horizontal in plane
				2.	| 0	| 1	| 0	Vertical in plane
				3.	| 0	| 0	| 1 Between planes
				"""

		return path, distance

	def longest_dimension_first_routing(self, source, destinations):
		distance = {source: t_router}
		path = {source: 'source'}

		for destination in destinations:
			delta_x, delta_y, delta_z = self.delta(source, destination)
			deltas = {
				'x': delta_x,
				'y': delta_y,
				'z': delta_z
			}

			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			while deltas:
				temp = {key: abs(item) for key, item in deltas.items()}
				direction = max(temp, key=temp.get)
				delta = deltas[direction]

				x, y, z = direction == 'x', direction == 'y', direction == 'z'

				while delta:
					next_node = \
						((current_node[0] + sign(delta) * x) % self.size[0],
						 (current_node[1] + sign(delta) * y) % self.size[1],
						 (current_node[2] + sign(delta) * z) % self.size[2])
					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'ldfr-routing calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'ldfr-routing tried to go from {current_node} to {next_node}, but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= sign(delta)

				del deltas[direction]

		return path, distance

	# Routing Sub-methods
	def delta(self, source, destination):
		delta = [destination[0] - source[0], destination[1] - source[1], destination[2] - source[2]]
		if self.torus:
			if abs(delta[0]) > 1 / 2 * self.size[0]:
				delta[0] -= sign(delta[0]) * self.size[0]
			if abs(delta[1]) > 1 / 2 * self.size[1]:
				delta[1] -= sign(delta[1]) * self.size[1]
			if abs(delta[2]) > 1 / 2 * self.size[2]:
				delta[2] -= sign(delta[2]) * self.size[2]

		return delta

	# Mapping Sub-methods
	def distance(self, node_A, node_B):
		# Calculates the distance between two nodes, The euclydean distance in this case
		deltas = self.delta(node_A, node_B)
		distance = (deltas[0]**2 + deltas[1]**2 + deltas[2]**2)**(1 / 2)
		return distance


class HubNetwork(Mesh4):
	def __init__(self, size, torus, link_length, topology = 'mesh4'):
		Network.__init__(self)
		self.torus = True
		if size % link_length:
			sim_log.notice(
				f'The given network size {size} is not a multiple of the link length {link_length}.\n'
				f'The network size is increased to {ceil(size / link_length) * link_length} '
				f'in order to achieve an evenly distribution of hubs.')
			size = ceil(size / link_length) * link_length
		if torus == 'False':
			sim_log.notice(
				'At this point in time, the HubNetwork class always assumes torus connections in the secondary level.\n'
				'Current run is set up with torus = False')
		sim_log.message('Create lower level Mesh4 hardware graph')
		self.size = size
		self.hubs = []
		self.link_length = link_length
		self.create_network(size)
		self.create_secondary_level(topology, link_length)
		self.sec_topology = topology
		self.type = 'Hub-Network'

	####################################################################################################################
	# Construction Methods
	####################################################################################################################
	def create_secondary_level(self, topology, weight = 1):
		size = self.size
		if topology.lower() == 'mesh4':
			sim_log.message('Create secondary level Mesh4-connections')

			# Create connection to all neighbours for each node
			for (x, y) in self.nodes.keys():
				if x % self.link_length == floor(self.link_length / 2) and y % self.link_length == floor(
						self.link_length / 2):
					self.add_edge((x, y), ((x + self.link_length) % size[0], y), weight)
					self.add_edge((x, y), (x, (y + self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x - self.link_length) % size[0], y), weight)
					self.add_edge((x, y), (x, (y - self.link_length) % size[1]), weight)
					self.hubs.append((x, y))
		elif topology.lower() == 'mesh6':
			sim_log.message('Create secondary level Mesh6-connections')

			# Create connection to all neighbours for each node
			for (x, y) in self.nodes.keys():
				if x % self.link_length == floor(self.link_length / 2) and y % self.link_length == floor(
						self.link_length / 2):
					self.add_edge((x, y), ((x + self.link_length) % size[0], y), weight)
					self.add_edge((x, y), (x, (y + self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x - self.link_length) % size[0], y), weight)
					self.add_edge((x, y), (x, (y - self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x + self.link_length) % size[0], (y + self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x - self.link_length) % size[0], (y - self.link_length) % size[1]), weight)
					self.hubs.append((x, y))
		elif topology.lower() == 'mesh8':
			sim_log.message('Create secondary level Mesh8-connections')

			# Create connection to all neighbours for each node
			for (x, y) in self.nodes.keys():
				if x % self.link_length == floor(self.link_length / 2) and y % self.link_length == floor(
						self.link_length / 2):
					self.add_edge((x, y), ((x + self.link_length) % size[0], y), weight)
					self.add_edge((x, y), (x, (y + self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x - self.link_length) % size[0], y), weight)
					self.add_edge((x, y), (x, (y - self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x + self.link_length) % size[0], (y + self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x - self.link_length) % size[0], (y - self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x + self.link_length) % size[0], (y - self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x - self.link_length) % size[0], (y + self.link_length) % size[1]), weight)
					self.hubs.append((x, y))
		elif topology.lower() == 'rotated grid':
			# TODO: Change, now we get two seperated secondary levels
			# TODO: Implement Rotated grid
			sim_log.message('Create secondary level 45 degrees rotated Mesh4-connections')

			# Create connection to all neighbours for each node
			for (x, y) in self.nodes.keys():
				if x % self.link_length == floor(self.link_length / 2) and y % self.link_length == floor(
						self.link_length / 2):
					self.add_edge((x, y), ((x + self.link_length) % size[0], (y + self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x + self.link_length) % size[0], (y - self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x - self.link_length) % size[0], (y - self.link_length) % size[1]), weight)
					self.add_edge((x, y), ((x - self.link_length) % size[0], (y + self.link_length) % size[1]), weight)
					self.hubs.append((x, y))

		sim_log.message(f'\tSecondary level generated.')

	####################################################################################################################
	# Factory Methods
	####################################################################################################################
	def routing_algorithm(self, routing_type):
		if routing_type.lower() == 'dor':
			return self.dimension_order_routing
		elif routing_type.lower() == 'ldfr':
			return self.longest_dimension_first_routing
		elif routing_type.lower() == 'spr' or routing_type.lower() == 'dijkstra':
			return self.dijkstra
		else:
			sim_log.error(
				f'Given routing algorithm {routing_type} is not recognized or implemented for Hub networks. '
				f'Continue with next iteration.')
			raise sim_log.RoutingError

	def population_mapping(self, mapping_type):
		if mapping_type.lower() == 'random':
			return self.random_population_mapping
		if mapping_type.lower() == 'sequential':
			return self.sequential_population_mapping
		if mapping_type.lower() == 'population_placement':
			return self.population_placement
		if mapping_type.lower() == 'clustered':
			return self.clustered_placement
		if mapping_type.lower() == '':
			pass
			#TODO: Add area clustered mapping grouping areas in clusters together

	####################################################################################################################
	# Casting Methods
	####################################################################################################################
	def broadcastfirst_unicast(self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations = {}
		number_of_spikes = 0
		destinations_node = {}

		routes_pre_phase3, distances_to_hubs = self.broadcastfirst_prephase(Node_ID, routing_fnc)

		# Determine destinations
		if netlist:
			for neuron in Node_object.neurons:
				# Convert combined connections list to weighted connections dictionary with unique entries
				if netlist[neuron]['FR'] == 0:
					continue
				destinations_neuron = \
					[netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				destinations_node[neuron] = destinations_neuron
				number_of_spikes += netlist[neuron]['FR']
				for target in destinations_neuron:
					destinations[target] = destinations.get(target, 0) + netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue

				for i in range(tally):
					destinations_neuron = self.destinations_matrix(population_index, matrix)

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = destinations_neuron
					number_of_spikes += matrix[population_index].get('FR', 1)
					for target in destinations_neuron:
						destinations[target] = destinations.get(target, 0) + matrix[population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		occupied_links, routes_phase3, distances = \
			self.broadcastfirst_postphase(
				destinations_node, routes_pre_phase3, number_of_spikes, distances_to_hubs, routing_fnc)

		for destination, firerate in destinations.items():
			for link in routes_phase3[destination]:
				occupied_links[link] = occupied_links.get(link, 0) + firerate

		for neuron, destinations_neuron in destinations_node.items():
			latency.update({
				neuron: max([distances[target_Node] for target_Node in set(destinations_neuron)])
			})

		return Node_ID, occupied_links, number_of_spikes, latency

	def broadcastfirst_local_multicast(
		self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations = {}
		number_of_spikes = 0
		destinations_node = {}

		routes_pre_phase3, distances_to_hubs = self.broadcastfirst_prephase(Node_ID, routing_fnc)

		# Determine destinations
		if netlist:
			for neuron in Node_object.neurons:
				# Convert combined connections list to weighted connections dictionary with unique entries
				if netlist[neuron]['FR'] == 0:
					continue
				destinations_neuron = \
					[netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				# Remove duplicate target nodes for a single neuron
				destinations_neuron = list(set(destinations_neuron))

				destinations_node[neuron] = destinations_neuron
				number_of_spikes += netlist[neuron]['FR']
				for target in destinations_neuron:
					destinations[target] = destinations.get(target, 0) + netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue

				for i in range(tally):
					destinations_neuron = self.destinations_nodes_matrix(population_index, matrix)

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = destinations_neuron
					number_of_spikes += matrix[population_index].get('FR', 1)
					for target in destinations_neuron:
						destinations[target] = destinations.get(target, 0) + matrix[
							population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		occupied_links, routes_phase3, distances = \
			self.broadcastfirst_postphase(
				destinations_node, routes_pre_phase3, number_of_spikes, distances_to_hubs, routing_fnc)

		if any([destination not in routes_phase3.keys() for destination in destinations.keys()]):
			print('\n\n')
			for dest in destinations.keys():
				if dest not in routes_phase3.keys():
					print(f'{dest} was not found in the routes_phase3 dictionary')

			print(time.ctime())

		for destination, firerate in destinations.items():
			for link in routes_phase3[destination]:
				occupied_links[link] = occupied_links.get(link, 0) + firerate

		for neuron, destinations_neuron in destinations_node.items():
			latency.update({
				neuron: max([distances[target_Node] for target_Node in set(destinations_neuron)])
			})

		return Node_ID, occupied_links, number_of_spikes, latency

	def broadcastfirst_multicast(self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations_node = {}
		firing_rate = {}

		routes_pre_phase3, distances_to_hubs = self.broadcastfirst_prephase(Node_ID, routing_fnc)

		# Determine destinations
		if netlist:
			for neuron in Node_object.neurons:
				if netlist[neuron]['FR'] == 0:
					continue
				destinations = \
					[netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				# Remove duplicate target nodes
				destinations = list(set(destinations))
				destinations_node[neuron] = destinations
				firing_rate[neuron] = netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue
				for i in range(tally):
					destinations = self.destinations_nodes_matrix(population_index, matrix)

					# Remove duplicate target nodes
					destinations = list(set(destinations))

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = destinations
					firing_rate[neuron_ID] = matrix[population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		number_of_spikes = sum(firing_rate.values())

		occupied_links, routes_phase3, distances = \
			self.broadcastfirst_postphase(
				destinations_node, routes_pre_phase3, number_of_spikes, distances_to_hubs, routing_fnc)

		for neuron, destinations_neuron in destinations_node.items():
			neuron_route = []
			for destination in destinations_neuron:
				neuron_route += routes_phase3[destination]

			# Remove duplicates
			neuron_route = list(set(neuron_route))

			for link in neuron_route:
				occupied_links[link] = occupied_links.get(link, 0) + firing_rate[neuron]

			latency[neuron] = max([distances[target_Node] for target_Node in destinations_neuron])

		return Node_ID, occupied_links, number_of_spikes, latency

	def flood(
		self, Node_ID, Node_object, routing_type, casting, netlist = None, matrix = None, flooding_routes = None):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations_node = {}  # only store the hubs which cover at least one destination node
		number_of_spikes = 0
		firing_rate = {}
		occupied_links = {}

		# Phase 0: Determine all hubs that cover at least one destination node
		if netlist:
			for neuron in Node_object.neurons:
				# Convert combined connections list to weighted connections dictionary with unique entries
				if netlist[neuron]['FR'] == 0:
					continue
				destinations = [netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				# Remove duplicate target nodes
				destinations = list(set(destinations))
				destinations_node[neuron] = list(set([self.closest_hub(destination) for destination in destinations]))
				firing_rate[neuron] = netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue
				for i in range(tally):
					destinations = self.destinations_nodes_matrix(population_index, matrix)

					# Remove duplicate target nodes
					destinations = list(set(destinations))

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = list(
						set([self.closest_hub(destination) for destination in destinations]))
					firing_rate[neuron_ID] = matrix[population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		# Phase 1: Route to closest hub
		route_to_closest_hub, distance_to_closest_hub, start_hub = self.route_to_closest_hub(Node_ID, routing_fnc)

		for neuron, destinations_neuron in destinations_node.items():
			# Phase 2: Route to the hubs of interest
			paths, distances = self.secondary_layer_routing(start_hub, destinations_neuron, distance_to_closest_hub)
			routes_to_hubs = []
			flooding = []
			for destination in destinations_neuron:
				current_node = destination
				while paths[current_node] != 'hub':
					routes_to_hubs.append((paths[current_node], current_node))
					current_node = paths[current_node]

				# Phase 3: Flood/Broadcast corresponding hubs
				flooding += flooding_routes[0][destination]

			if casting == 'mc':
				routes_to_hubs = list(set(routes_to_hubs)) + route_to_closest_hub
				number_of_spikes += firing_rate[neuron]
			elif casting == 'uc':
				routes_to_hubs = routes_to_hubs + route_to_closest_hub
				number_of_spikes += firing_rate[neuron]

			for link in routes_to_hubs + flooding:
				occupied_links[link] = occupied_links.get(link, 0) + firing_rate[neuron]

			latency[neuron] = max(
				[distances[destination] + flooding_routes[1][destination] for destination in destinations_neuron])

		return Node_ID, occupied_links, number_of_spikes, latency

	def uc_flood(
		self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, flooding_routes = None):
		return self.flood(Node_ID, Node_object, routing_type, 'uc', netlist, matrix, flooding_routes)

	def mc_flood(
		self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, flooding_routes = None):
		return self.flood(Node_ID, Node_object, routing_type, 'mc', netlist, matrix, flooding_routes)

	# Casting sub-Methods
	def broadcastfirst_prephase(self, Node_ID, routing_fnc):
		# Phase 1 of casting: move toward the nearest hub
		route_to_closest_hub, distance_to_closest_hub, start_hub = self.route_to_closest_hub(Node_ID, routing_fnc)
		# Phase 2 of casting: from nearest hub broadcast to all other hubs
		route_to_hubs, distances_to_hubs = self.broadcast_hubs(start_hub, distance_to_closest_hub)

		# The combined routes from phase 1 and 2 are used once for every spike send out from this node
		routes_pre_phase = route_to_closest_hub + route_to_hubs

		return routes_pre_phase, distances_to_hubs

	def broadcastfirst_postphase(
		self, destinations_node, routes_pre_phase, number_of_spikes, distances_to_hubs, routing_fnc):
		occupied_links = {}
		for link in routes_pre_phase:
			occupied_links[link] = number_of_spikes

		# Phase 3 of casting: From the hubs to all surround destination nodes
		all_destinations = list(set(
			[destination for dest_list in destinations_node.values() for destination in dest_list]))
		routes_phase3, distances = self.route_hubs_to_nodes(all_destinations, distances_to_hubs, routing_fnc)

		return occupied_links, routes_phase3, distances

	def flood_hubs(self):
		visited_nodes = []
		flood_routes = {}
		distances = {}

		for node in self.nodes:
			if node not in visited_nodes:
				source_hub = self.closest_hub(node)
				node_to_hub, distance_from_hub = self.longest_dimension_first_routing(source_hub, [node])

				current_node = node
				route = []
				while node_to_hub[current_node] != 'source' and current_node not in visited_nodes:
					route.append((node_to_hub[current_node], current_node))
					visited_nodes.append(current_node)
					current_node = node_to_hub[current_node]

				flood_routes[source_hub] = flood_routes.get(source_hub, []) + route

				if distances.get(source_hub, 0) < distance_from_hub[node] - t_router:
					distances[source_hub] = distance_from_hub[node] - t_router

		return flood_routes, distances

	def broadcast_hubs(self, nearest_hub, distance_to_hub):
		paths_hubs, distance = self.secondary_layer_routing(nearest_hub, self.hubs, distance_to_hub)
		del paths_hubs[nearest_hub]
		routes_hubs = []
		for current_node, previous_node in paths_hubs.items():
			routes_hubs.append((previous_node, current_node))

		return routes_hubs, distance

	####################################################################################################################
	# Routing Methods
	####################################################################################################################
	def secondary_layer_routing(self, source, destinations, dist):  # Fixed at LDFR algorithm
		if self.sec_topology.lower() == 'mesh4':
			return self.secondary_Mesh4_routing(source, destinations, dist)
		elif self.sec_topology.lower() == 'mesh6':
			return self.secondary_Mesh6_routing(source, destinations, dist)
		elif self.sec_topology.lower() == 'mesh8':
			return self.secondary_Mesh8_routing(source, destinations, dist)
		elif self.sec_topology.lower() == 'rotated grid':
			return self.secondary_rotated_grid_routing(source, destinations, dist)

	def secondary_Mesh4_routing(self, source, destinations, dist):
		distance = {source: dist}
		path = {source: 'hub'}
		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			delta_x, delta_y = self.delta(source, destination)
			deltas = {
				'x': delta_x,
				'y': delta_y
			}

			while deltas:
				temp = {key: abs(item) for key, item in deltas.items()}
				direction = max(temp, key=temp.get)
				delta = deltas[direction]

				x, y = direction == 'x', direction == 'y'

				while delta:
					next_node = \
						((current_node[0] + self.link_length * sign(delta) * x) % self.size[0],
						 (current_node[1] + self.link_length * sign(delta) * y) % self.size[1])
					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'Routing in secondary level calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'Routing in secondary level tried to go from {current_node} to {next_node}, '
							f'but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= self.link_length * sign(delta)

				del deltas[direction]

		return path, distance

	def secondary_Mesh6_routing(self, source, destinations, dist):
		distance = {source: dist}
		path = {source: 'hub'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			delta_x, delta_y = self.delta(source, destination)

			if delta_x * delta_y >= 0:  # Upper right or lower left quadrant
				delta_w = min((delta_x, delta_y), key=abs)
				delta_x, delta_y, delta_w = [delta_x - delta_w, delta_y - delta_w, delta_w]

			elif self.torus and \
				abs(delta_x) + abs(delta_y) > min(self.size[0] - abs(delta_x), self.size[1] - abs(delta_y)):
				if self.size[0] - abs(delta_x) < self.size[1] - abs(delta_y):
					delta_x = delta_x - sign(delta_x) * self.size[0]
				else:
					delta_y = delta_y - sign(delta_y) * self.size[1]

				delta_w = min((delta_x, delta_y), key=abs)

				delta_x, delta_y, delta_w = [delta_x - delta_w, delta_y - delta_w, delta_w]
			else:
				delta_x, delta_y, delta_w = [delta_x, delta_y, 0]

			deltas = {
				'x': delta_x,
				'y': delta_y,
				'w': delta_w
			}

			while deltas:
				temp = {key: abs(item) for key, item in deltas.items()}
				direction = max(temp, key=temp.get)
				delta = deltas[direction]

				if direction == 'w':
					x, y = True, True
				else:
					x, y = direction == 'x', direction == 'y'

				while delta:
					next_node = \
						((current_node[0] + self.link_length * sign(delta) * x) % self.size[0],
						 (current_node[1] + self.link_length * sign(delta) * y) % self.size[1])
					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'Routing in secondary level calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'Routing in secondary level tried to go from {current_node} to {next_node}, '
							f'but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= self.link_length * sign(delta)

				del deltas[direction]

		return path, distance

	def secondary_Mesh8_routing(self, source, destinations, dist):
		distance = {source: dist}
		path = {source: 'hub'}

		for destination in destinations:
			if destination in path:
				# Destination has already been visited during a prior calculation, continue to next destination
				continue
			else:
				current_node = source

			delta_x, delta_y = self.delta(source, destination)

			if delta_x * delta_y >= 0:  # Move Diagonally
				delta_w = min((delta_x, delta_y), key=abs)
				delta_x, delta_y, delta_w, delta_v = [delta_x - delta_w, delta_y - delta_w, delta_w, 0]
			else:  # move diagonal from the top left to bottom right
				delta_v = abs(min((delta_x, delta_y), key=abs)) * sign(delta_x)
				delta_x, delta_y, delta_w, delta_v = [delta_x - delta_v, delta_y + delta_v, 0, delta_v]

			deltas = {
				'x': delta_x,
				'y': delta_y,
				'w': delta_w,
				'v': delta_v
			}

			while deltas:
				temp = {key: abs(item) for key, item in deltas.items()}
				direction = max(temp, key=temp.get)
				delta = deltas[direction]

				if direction == 'x':
					x, y = 1, 0
				elif direction == 'y':
					x, y = 0, 1
				elif direction == 'w':
					x, y = 1, 1
				else:
					x, y = 1, -1

				while delta:
					next_node = \
						((current_node[0] + self.link_length * sign(delta) * x) % self.size[0],
						 (current_node[1] + self.link_length * sign(delta) * y) % self.size[1])
					if next_node not in self.nodes:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'Routing in secondary level calculated the next node: {next_node}, but this node does not exist.')
						raise sim_log.RoutingError()
					elif next_node not in self.nodes[current_node].edges:
						sim_log.error(
							f'While routing from {source} to {destination}, '
							f'Routing in secondary level tried to go from {current_node} to {next_node}, '
							f'but no connection exists between these nodes.')
						raise sim_log.RoutingError()
					else:
						path[next_node] = current_node
						distance[next_node] = \
							distance[current_node] + t_router + self.nodes[current_node].edges[
								next_node].weight * t_link
						current_node = next_node
						delta -= self.link_length * sign(delta)

				del deltas[direction]

		return path, distance

	def secondary_rotated_grid_routing(self, source, destinations, dist):
		# TODO: Implement Rotated grid
		pass

	# Routing Sub-Methods
	def closest_hub(self, Node):
		x, y = Node
		Hx = floor(x / self.link_length) * self.link_length + floor(self.link_length / 2)
		Hy = floor(y / self.link_length) * self.link_length + floor(self.link_length / 2)

		return Hx, Hy

	def route_to_closest_hub(self, Node, routing_function):
		closest_hub = self.closest_hub(Node)
		path_h, distance_h = routing_function(Node, [closest_hub])
		distance_to_hub = distance_h[closest_hub] * t_router + distance_h[closest_hub] * t_link
		route_to_hub = []
		current_node = closest_hub
		while path_h[current_node] != 'source':
			route_to_hub.append((path_h[current_node], current_node))
			current_node = path_h[current_node]

		return route_to_hub, distance_to_hub, closest_hub

	def route_hubs_to_nodes(self, destinations, distances_to_hubs, routing_function):
		distance = distances_to_hubs
		closest_hubs = {}
		routes = {}
		for destination in destinations:
			closest_hub = self.closest_hub(destination)
			closest_hubs[closest_hub] = closest_hubs.get(closest_hub, []) + [destination]

		for Hub, local_destinations in closest_hubs.items():
			path_from_hub, distance_from_hub = routing_function(Hub, local_destinations)

			for destination in local_destinations:
				routes[destination] = []
				current_node = destination
				while current_node != Hub:
					previous_node = path_from_hub[current_node]
					routes[destination].append((previous_node, current_node))
					current_node = previous_node

				distance[destination] = distance[Hub] + distance_from_hub[destination] - t_router
		return routes, distance

	####################################################################################################################
	# Mapping Methods
	####################################################################################################################
	def cluster_placement(self, NpN, matrix, **aux_args):
		sim_log.message('Determining mapping of neurons... (Cluster placement)')
		T0 = time.time()
		network_size = sum([population["neurons"] for population in matrix])
		ratio = 0
		neurons_placed = 0
		print(f'Mapping neurons to network\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

		Nodes = list(self.nodes.keys())
		current_node = Nodes.pop(0)
		mapped_to_node = 0

		for population in matrix:
			population_name = population['population']
			population_size = population['neurons']
			neuron_counter = 0

			try:
				while neuron_counter < population_size:
					if mapped_to_node < NpN:
						tally = min(population_size - neuron_counter, NpN - mapped_to_node)
						self.nodes[current_node].neurons.append((tally, population_name))
						mapped_to_node += tally
						neuron_counter += tally
						neurons_placed += tally
						if round(neurons_placed / network_size * 1000) / 10 > ratio:
							ratio = round(neurons_placed / network_size * 1000) / 10
							print(
								f'Mapping neurons randomly to network\t| {" " * (ratio < 10)}{ratio} % |', end='\r')

					else:
						check_sum = sum([tally for tally, pop in self.nodes[current_node].neurons])
						if check_sum != NpN:
							sim_log.error(
								f'Expected number of neurons mapped to node {current_node} is {mapped_to_node}.\n'
								f'Actual number = {check_sum}.\n'
								f'Node contains the following neurons: {self.nodes[current_node].neurons}')
							raise sim_log.MappingError

						current_node = Nodes.pop(0)
						mapped_to_node = 0
			except IndexError:
				sim_log.error(
					f'All nodes have been filled.\n{neurons_placed} neurons have been placed, '
					f'{network_size - neurons_placed} have not.')
				raise sim_log.MappingError()

		sim_log.message(
			f'Mapped {network_size} neurons to hardware in {convert_time(time.time() - T0)}.\n'
			f'Hardware utilization: {round(network_size / (len(self.nodes) * NpN) * 1000) / 10}%\n')

	####################################################################################################################
	# Output Methods
	####################################################################################################################
	def readout(self, unused, compressed = None):
		output = {}
		int_lst = []
		ext_lst = []
		router_total = []
		link_lst1 = []
		link_lst2 = []
		for node in self.nodes.keys():
			output[str(node)] = {
				'int_packets_handled': self.nodes[node].int_packets_handled,
				'ext_packets_handled': self.nodes[node].ext_packets_handled,
				'edges': {}
			}
			if self.nodes[node].int_packets_handled or not unused:
				int_lst.append(self.nodes[node].int_packets_handled)
			if self.nodes[node].ext_packets_handled or not unused:
				ext_lst.append(self.nodes[node].ext_packets_handled)
			if (self.nodes[node].ext_packets_handled or self.nodes[node].int_packets_handled) or not unused:
				router_total.append(self.nodes[node].int_packets_handled + self.nodes[node].ext_packets_handled)
			for key, edge in self.nodes[node].edges.items():
				output[str(node)]['edges'][str(key)] = {'length': edge.weight, 'packets_handled': edge.packets_handled}
				if not unused or edge.packets_handled or self.nodes[key].edges[node].packets_handled:
					if edge.weight == 1:
						link_lst1.append(edge.packets_handled)
					else:
						link_lst2.append(edge.packets_handled)

		output['int_packets_handled'] = {
			'average': statistics.mean(int_lst),
			'min': min(int_lst),
			'max': max(int_lst),
			'median': statistics.median(int_lst)
		}
		output['ext_packets_handled'] = {
			'average': statistics.mean(ext_lst),
			'min': min(ext_lst),
			'max': max(ext_lst),
			'median': statistics.median(ext_lst)
		}
		output['packets_handled_per_node'] = {
			'average': statistics.mean(router_total),
			'min': min(router_total),
			'max': max(router_total),
			'median': statistics.median(router_total)
		}
		output['spikes_per_link_primary_layer'] = {
			'average': statistics.mean(link_lst1),
			'min': min(link_lst1),
			'max': max(link_lst1),
			'median': statistics.median(link_lst1)
		}
		output['spikes_per_link_secondary_layer'] = {
			'average': statistics.mean(link_lst2),
			'min': min(link_lst2),
			'max': max(link_lst2),
			'median': statistics.median(link_lst2)
		}

		if not compressed:
			return output
		elif compressed.lower() == 'node':
			return router_total
		elif compressed.lower() == 'link':
			return link_lst1


class HubNetwork_BC(Mesh4):
	def __init__(self, size, torus, N0, RF_weight = 1):
		Network.__init__(self)
		self.torus = True
		if torus == 'False':
			sim_log.notice(
				'At this point in time, the HubNetwork class always assumes torus connections in the secondary level.\n'
				'Current run is set up with torus = False')
		sim_log.message('Create lower level Mesh4 hardware graph')
		self.size = size
		self.N0 = N0
		self.spectrum = self.Edge(RF_weight)
		self.hubs = []
		self.create_network(size)
		self.create_secondary_level()
		self.type = 'Hub-Network'

	def routing_algorithm(self, routing_type):
		if routing_type.lower() == 'dor':
			return self.dimension_order_routing
		elif routing_type.lower() == 'ldfr':
			return self.longest_dimension_first_routing
		elif routing_type.lower() == 'spr' or routing_type.lower() == 'dijkstra':
			return self.dijkstra
		else:
			sim_log.error(
				f'Given routing algorithm {routing_type} is not recognized or implemented for Hub networks. '
				f'Continue with next iteration.')
			raise sim_log.RoutingError

	def create_secondary_level(self):
		# Create connection to all neighbours for each node
		for (x, y) in self.nodes.keys():
			if x % self.N0 == floor(self.N0 / 2) and y % self.N0 == floor(
					self.N0 / 2):
				self.hubs.append((x, y))
				self.nodes[(x, y)].edges['Spectrum'] = self.spectrum

		sim_log.message(f'\tSecondary level generated.')

	def closest_hub(self, Node):
		x, y = Node
		Hx = floor(x / self.N0) * self.N0 + floor(self.N0 / 2)
		Hy = floor(y / self.N0) * self.N0 + floor(self.N0 / 2)

		return Hx, Hy


	####################################################################################################################
	# Casting Methods
	####################################################################################################################
	def broadcastfirst_unicast(self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations = {}
		number_of_spikes = 0
		destinations_node = {}

		routes_pre_phase, distances_to_hubs = self.broadcastfirst_prephase(Node_ID, routing_fnc)

		# Determine destinations
		if netlist:
			for neuron in Node_object.neurons:
				# Convert combined connections list to weighted connections dictionary with unique entries
				if netlist[neuron]['FR'] == 0:
					continue
				destinations_neuron = \
					[netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				destinations_node[neuron] = destinations_neuron
				number_of_spikes += netlist[neuron]['FR']
				for target in destinations_neuron:
					destinations[target] = destinations.get(target, 0) + netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue

				for i in range(tally):
					destinations_neuron = self.destinations_matrix(population_index, matrix)

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = destinations_neuron
					number_of_spikes += matrix[population_index].get('FR', 1)
					for target in destinations_neuron:
						destinations[target] = destinations.get(target, 0) + matrix[population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		occupied_links, routes_phase3, distances = \
			self.broadcastfirst_postphase(
				destinations_node, routes_pre_phase, number_of_spikes, distances_to_hubs, routing_fnc)

		for destination, firerate in destinations.items():
			for link in routes_phase3[destination]:
				occupied_links[link] = occupied_links.get(link, 0) + firerate

		for neuron, destinations_neuron in destinations_node.items():
			latency.update({
				neuron: max([distances[target_Node] for target_Node in set(destinations_neuron)])
			})

		return Node_ID, occupied_links, number_of_spikes, latency

	def broadcastfirst_local_multicast(
		self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations = {}
		number_of_spikes = 0
		destinations_node = {}

		routes_pre_phase, distances_to_hubs = self.broadcastfirst_prephase(Node_ID, routing_fnc)

		# Determine destinations
		if netlist:
			for neuron in Node_object.neurons:
				# Convert combined connections list to weighted connections dictionary with unique entries
				if netlist[neuron]['FR'] == 0:
					continue
				destinations_neuron = \
					[netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				# Remove duplicate target nodes for a single neuron
				destinations_neuron = list(set(destinations_neuron))

				destinations_node[neuron] = destinations_neuron
				number_of_spikes += netlist[neuron]['FR']
				for target in destinations_neuron:
					destinations[target] = destinations.get(target, 0) + netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue

				for i in range(tally):
					destinations_neuron = self.destinations_nodes_matrix(population_index, matrix)

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = destinations_neuron
					number_of_spikes += matrix[population_index].get('FR', 1)
					for target in destinations_neuron:
						destinations[target] = destinations.get(target, 0) + matrix[
							population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		occupied_links, routes_phase3, distances = \
			self.broadcastfirst_postphase(
				destinations_node, routes_pre_phase, number_of_spikes, distances_to_hubs, routing_fnc)
		try:
			for destination, firerate in destinations.items():
				for link in routes_phase3[destination]:
					occupied_links[link] = occupied_links.get(link, 0) + firerate
		except KeyError:
			for dest in destinations.keys():
				print(f'{dest}: ', end='')
				print(routes_phase3)

		for neuron, destinations_neuron in destinations_node.items():
			latency.update({
				neuron: max([distances[target_Node] for target_Node in set(destinations_neuron)])
			})

		return Node_ID, occupied_links, number_of_spikes, latency

	def broadcastfirst_multicast(self, Node_ID, Node_object, routing_type, netlist = None, matrix = None, **kargs):
		routing_fnc = self.routing_algorithm(routing_type)
		latency = {}
		destinations_node = {}
		firing_rate = {}

		routes_pre_phase, distances_to_hubs = self.broadcastfirst_prephase(Node_ID, routing_fnc)

		# Determine destinations
		if netlist:
			for neuron in Node_object.neurons:
				if netlist[neuron]['FR'] == 0:
					continue
				destinations = \
					[netlist[connection]['location'] for connection in netlist[neuron]['connected_to']]

				# Remove duplicate target nodes
				destinations = list(set(destinations))
				destinations_node[neuron] = destinations
				firing_rate[neuron] = netlist[neuron]['FR']
		elif matrix:
			populations = [item['population'] for item in matrix]
			for (tally, population) in Node_object.neurons:
				population_index = populations.index(population)
				if matrix[population_index].get('FR', 1) == 0:
					continue
				for i in range(tally):
					destinations = self.destinations_nodes_matrix(population_index, matrix)

					# Remove duplicate target nodes
					destinations = list(set(destinations))

					neuron_ID = f'{Node_ID}_{population}_{i}'
					destinations_node[neuron_ID] = destinations
					firing_rate[neuron_ID] = matrix[population_index].get('FR', 1)
		else:
			sim_log.error(
				f'Either a NN netlist or a connectivity probability matrix is required to run the simulation.\n'
				f'Abort!')
			raise sim_log.SimError

		number_of_spikes = sum(firing_rate.values())

		occupied_links, routes_phase3, distances = \
			self.broadcastfirst_postphase(
				destinations_node, routes_pre_phase, number_of_spikes, distances_to_hubs, routing_fnc)

		for neuron, destinations_neuron in destinations_node.items():
			neuron_route = []
			for destination in destinations_neuron:
				neuron_route += routes_phase3[destination]

			# Remove duplicates
			neuron_route = list(set(neuron_route))

			for link in neuron_route:
				occupied_links[link] = occupied_links.get(link, 0) + firing_rate[neuron]

			latency[neuron] = max([distances[target_Node] for target_Node in destinations_neuron])

		return Node_ID, occupied_links, number_of_spikes, latency

	# Casting Sub-Methods
	def broadcastfirst_prephase(self, Node_ID, routing_fnc):
		# Phase 1 of casting: move toward the nearest hub
		route_to_closest_hub, distance_to_closest_hub, start_hub = self.route_to_closest_hub(Node_ID, routing_fnc)
		distances_to_hubs = {Node_ID: distance_to_closest_hub}

		# The combined routes from phase 1 and 2 are used once for every spike send out from this node
		routes_pre_phase = route_to_closest_hub + [(self.closest_hub(Node_ID), 'Spectrum')]
		for hub in self.hubs:
			distances_to_hubs[hub] = distance_to_closest_hub + self.spectrum.weight

		return routes_pre_phase, distances_to_hubs

	def broadcastfirst_postphase(
		self, destinations_node, routes_pre_phase3, number_of_spikes, distances_to_hubs, routing_fnc):
		occupied_links = {}
		for link in routes_pre_phase3:
			occupied_links[link] = number_of_spikes

		# Phase 3 of casting: From the hubs to all surround destination nodes
		all_destinations = list(set(
			[destination for dest_list in destinations_node.values() for destination in dest_list]))
		routes_phase3, distances = self.route_hubs_to_nodes(all_destinations, distances_to_hubs, routing_fnc)

		return occupied_links, routes_phase3, distances

	####################################################################################################################
	# Routing Methods
	####################################################################################################################
	def route_hubs_to_nodes(self, destinations, distances_to_hubs, routing_function):
		distance = distances_to_hubs
		closest_hubs = {}
		routes = {}
		for destination in destinations:
			closest_hub = self.closest_hub(destination)
			closest_hubs[closest_hub] = closest_hubs.get(closest_hub, []) + [destination]

		for Hub, local_destinations in closest_hubs.items():
			path_from_hub, distance_from_hub = routing_function(Hub, local_destinations)

			for destination in local_destinations:
				routes[destination] = []
				current_node = destination
				while current_node != Hub:
					previous_node = path_from_hub[current_node]
					routes[destination].append((previous_node, current_node))
					current_node = previous_node

				distance[destination] = distance[Hub] + distance_from_hub[destination] - t_router
		return routes, distance

	def route_to_closest_hub(self, Node, routing_function):
		closest_hub = self.closest_hub(Node)
		path_h, distance_h = routing_function(Node, [closest_hub])
		distance_to_hub = distance_h[closest_hub] * t_router + distance_h[closest_hub] * t_link
		route_to_hub = []
		current_node = closest_hub
		while path_h[current_node] != 'source':
			route_to_hub.append((path_h[current_node], current_node))
			current_node = path_h[current_node]

		return route_to_hub, distance_to_hub, closest_hub


	####################################################################################################################
	# Output Methods
	####################################################################################################################
	def readout(self, unused, compressed = None):
		output = {}
		int_lst = []
		ext_lst = []
		router_total = []
		link_lst = []
		for node in self.nodes.keys():
			output[str(node)] = {
				'int_packets_handled': self.nodes[node].int_packets_handled,
				'ext_packets_handled': self.nodes[node].ext_packets_handled,
				'edges': {}
			}
			if self.nodes[node].int_packets_handled or not unused:
				int_lst.append(self.nodes[node].int_packets_handled)
			if self.nodes[node].ext_packets_handled or not unused:
				ext_lst.append(self.nodes[node].ext_packets_handled)
			if (self.nodes[node].ext_packets_handled or self.nodes[node].int_packets_handled) or not unused:
				router_total.append(self.nodes[node].int_packets_handled + self.nodes[node].ext_packets_handled)
			output['Spectrum'] = {'Delay': self.spectrum.weight, 'Bandwidth': self.spectrum.packets_handled}
			for key, edge in self.nodes[node].edges.items():
				if not key == 'Spectrum':
					output[str(node)]['edges'][str(key)] = {
						'length': edge.weight, 'packets_handled': edge.packets_handled}

				if not unused or edge.packets_handled or self.nodes[key].edges[node].packets_handled:
					link_lst.append(edge.packets_handled)

		output['int_packets_handled'] = {
			'average': statistics.mean(int_lst),
			'min': min(int_lst),
			'max': max(int_lst),
			'median': statistics.median(int_lst)
		}
		output['ext_packets_handled'] = {
			'average': statistics.mean(ext_lst),
			'min': min(ext_lst),
			'max': max(ext_lst),
			'median': statistics.median(ext_lst)
		}
		output['packets_handled_per_node'] = {
			'average': statistics.mean(router_total),
			'min': min(router_total),
			'max': max(router_total),
			'median': statistics.median(router_total)
		}
		output['spikes_per_link_primary_layer'] = {
			'average': statistics.mean(link_lst),
			'min': min(link_lst),
			'max': max(link_lst),
			'median': statistics.median(link_lst)
		}

		if not compressed:
			return output
		elif compressed.lower() == 'node':
			return router_total
		elif compressed.lower() == 'link':
			return link_lst
