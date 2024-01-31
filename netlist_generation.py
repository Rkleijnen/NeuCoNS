"""
netlist_generation.py

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

import random
import os
import json
import sim_log


########################################################################################################################
# Factory Methods
########################################################################################################################
def netlist_factory(testcase):
	if testcase.lower() == 'hh' or testcase.lower() == 'hopfield':
		generator = Hopfield_NN
	elif testcase.lower() == 'rndc':
		generator = randomly_connected_NN
	elif testcase.lower() == 'vogelsabbott' or testcase.lower() == 'vogels':
		generator = TwoPopulationNetwork_VogelsAbbott
	elif testcase.lower() == 'brunel':
		generator = TwoPopulationNetwork_Brunel
	elif testcase.lower() == 'potjansdiesmann' or testcase.lower() == 'potjans_diesmann' \
		or testcase.lower() == 'corticalmicrocircuit' or testcase.lower() == 'cortical_microcircuit':
		generator = CorticalMicrocircuit_PotjansDiesmann
	elif testcase.lower() == 'cm_benchmark'\
		or testcase.lower() == 'corticalmicrocircuit_benchmark' or testcase.lower() == 'cortical_microcircuit_benchmark':
		generator = CorticalMicrocircuit_Benchmark
	elif testcase.lower() == 'file':
		generator = read_netlist_file
	else:
		sim_log.fatal_error(f"Testcase type {testcase} not defined")
		generator = None
	return generator


########################################################################################################################
# Read functions
########################################################################################################################
def read_netlist_file(file_name = ''):
	if not file_name:
		sim_log.fatal_error(
			f'No netlist file given.'
			f'Abort run.'
		)
	elif os.path.exists(file_name):
		with open(file_name, 'r') as file:
			netlist = json.load(file)
		sim_log.message(
			f'Netlist file: {file_name} Loaded.')
	else:
		sim_log.fatal_error(
			f'Could not find "{file_name}".'
			f'Abort run.'
		)

	return netlist, file_name


# If FR_defined=True, the third column of the matrix is read as the firing rate of the populations
def read_connectivity_probability_matrix(file_name, delimiter='\t'):
	file_name = file_name
	connectivity_matrix = []
	with open(file_name, 'r') as matrix_file:
		lines = matrix_file.readlines()
		if len(lines) + 3 == len(lines[0].split(delimiter)):
			FR_defined = True
			sim_log.message(f'Matrix file loaded with specified firing rates.')
		elif len(lines) + 2 == len(lines[0].split(delimiter)):
			FR_defined = False
			sim_log.message(f'Matrix file loaded without specified firing rates.')
		else:
			sim_log.fatal_error(
				f'The format of the matrix-file looks incorrect.\n'
				f'{len(lines)}')
		for line in lines:
			try:
				line = line.strip('\n').split(delimiter)
				if FR_defined:
					connectivity_matrix.append(
						{
							'population': line[0].strip(),
							'neurons': int(line[1]),
							'FR': float(line[2]),
							'connectivity_prob': [float(x) for x in line[3:]]
						})
				else:
					connectivity_matrix.append(
						{
							'population': line[0].strip(),
							'neurons': int(line[1]),
							'connectivity_prob': [float(x) for x in line[2:]]
						})

			except ValueError:
				sim_log.fatal_error(
					f'Error in line {lines.index(line) + 1}:\n'
					f'\t {line}')

	return connectivity_matrix


########################################################################################################################
# Neural network netlist generator functions
########################################################################################################################
# For the testcases in which the firing-rate has not (jet) been defined, a firing-rate of 1 is assumed
def randomly_connected_NN(n = 1000, epsilon = 0.1, netlist_name = None, path = ''):
	if not netlist_name:
		file_name = f'RNDC_N{n}_eps{epsilon}'
	else:
		file_name = netlist_name
	if os.path.exists(path + file_name + '.json'):
		with open(path + file_name + '.json', 'r') as file:
			netlist = json.load(file)
		sim_log.message(
			f'The simulation folder already contains a netlist file: {file_name}.json\nLoad existing file.')

		if not len(netlist) == n:
			sim_log.fatal_error(
				f'Loaded netlist contains {len(netlist)} neurons, this does not match the specified netlist settings (n := {n}).'
				f'\nNetlist file might be corrupted.'
				f'Rename or remove existing file and restart simulator to force generation a new netlist file.')
		return netlist, path + file_name + '.json'

	netlist = {}
	nr_synapses = 0
	for neuron in range(n):
		connectionlist = []
		for target_neuron in range(n):
			if random.random() <= epsilon:
				connectionlist.append(str(target_neuron))
				nr_synapses += 1

		netlist[str(neuron)] = {
			'FR': 1,
			'connected_to': connectionlist
		}

	sim_log.message(
		f'Generated netlist for randomly connected neural network:\n'
		f'\tNumber of neurons = {n}\n\tConnection probability = {epsilon}\n\tNr. of synapses = {nr_synapses}\n'
		f'Save netlist in {os.getcwd() + file_name}.json')

	if path:
		with open(path + file_name + '.json', 'w+') as file:
			json.dump(netlist, file)

	return netlist, path + file_name + '.json'


def randomly_connected_NN_C(n = 1000, connections = 100, netlist_name = None, path = ''):
	if not netlist_name:
		file_name = f'RNDC_N{n}_C{connections}'
	else:
		file_name = netlist_name

	if os.path.exists(path + file_name + '.json'):
		with open(path + file_name + '.json', 'r') as file:
			netlist = json.load(file)
		sim_log.message(
			f'The simulation folder already contains a netlist file: {file_name}.json\nLoad existing file.')

		if not len(netlist) == n:
			sim_log.fatal_error(
				f'Loaded netlist contains {len(netlist)} neurons, this does not match the specified netlist settings (n := {n}).'
				f'\nNetlist file might be corrupted.'
				f'Rename or remove existing file and restart simulator to force generation a new netlist file.')
		return netlist, path + file_name + '.json'

	netlist = {}
	nr_synapses = 0
	for neuron in range(n):
		connectionlist = []
		for c in range(connections):
			target_neuron = random.choice(list(range(n)))
			connectionlist.append(str(target_neuron))
			nr_synapses += 1

		netlist[str(neuron)] = {
			'FR': 1,
			'connected_to': connectionlist
		}

	sim_log.message(
		f'Generated netlist for randomly connected neural network:\n'
		f'\tNumber of neurons = {n}\n\tConnection probability = {connections}\n\tNr. of synapses = {nr_synapses}\n'
		f'Save netlist in {os.getcwd() + file_name}.json')
	if path:
		with open(path + file_name + '.json', 'w+') as file:
			json.dump(netlist, file)

	return netlist, path + file_name + '.json'


def Hopfield_NN(n = 1000, netlist_name = None, path = ''):
	if not netlist_name:
		file_name = f'Hopfield_N{n}'
	else:
		file_name = netlist_name

	if os.path.exists(path + file_name + '.json'):
		with open(path + file_name + '.json', 'r') as file:
			netlist = json.load(file)
		sim_log.message(
			f'The simulation folder already contains a netlist file: {file_name}.json\nLoad existing file.')

		if not len(netlist) == n:
			sim_log.fatal_error(
				f'Loaded netlist contains {len(netlist)} neurons, this does not match the specified netlist settings (n := {n}).'
				f'\nNetlist file might be corrupted.'
				f'Rename or remove existing file and restart simulator to force generation a new netlist file.')
		return netlist, path + file_name + '.json'

	netlist = {}
	connectionlist = [str(x) for x in range(n)]

	for neuron in range(n):
		netlist[str(neuron)] = {
			'FR': 1,
			'connected_to': connectionlist
		}

	sim_log.message(
		f'Generated netlist for Hopfield neural network:\n'
		f'\tNumber of neurons = {n}\n\tNr. of synapses = {n * n}\n'
		f'Save netlist in {os.getcwd() + file_name}.json')
	if path:
		with open(path + file_name + '.json', 'w+') as file:
			json.dump(netlist, file)

	return netlist, path + file_name + '.json'


def TwoPopulationNetwork_Brunel(n = 12500, beta = 0.8, epsilon = 0.1, netlist_name = None,  path = ''):
	# The network that is generated with this method does not exactly match the testcase provided in the IVF_ACA project.
	# The testcase is set to have a fixed in-degree and allows multapses (multiple synapses between 2 neurons)
	# In this method, the multapses are combined to a single synapse, which causes a reduction of the (fixed) in-degree.
	if not netlist_name:
		file_name = f'Brunel_N{n}_B{beta}_C{epsilon}'
	else:
		file_name = netlist_name

	if os.path.exists(path + file_name + '.json'):
		with open(path + file_name + '.json', 'r') as file:
			netlist = json.load(file)
		sim_log.message(
			f'The simulation folder already contains a netlist file: {file_name}.json\nLoad existing file.')

		if not len(netlist) == n:
			sim_log.fatal_error(
				f'Loaded netlist contains {len(netlist)} neurons, this does not match the specified netlist settings (n := {n}).'
				f'\nNetlist file might be corrupted.'
				f'Rename or remove existing file and restart simulator to force generation a new netlist file.')
		return netlist, path + file_name + '.json'

	Ne = int(beta * n)
	Ni = int(n - Ne)
	Ce = int(epsilon * Ne)
	Ci = int(epsilon * Ni)
	netlist = {}
	nr_synapses = 0

	for n in range(n):
		if n < Ne:
			netlist['E_' + str(n)] = {
				'FR': 1,
				'connected_to': []}
		else:
			netlist['I_' + str(n-Ne)] = {
				'FR': 1,
				'connected_to': []}

	for j in range(Ne):
		for _ in range(Ce):
			i = int(random.random() * n)
			if i < Ne:
				if ('E_' + str(j)) in netlist['E_' + str(i)]['connected_to']:
					continue
				netlist['E_' + str(i)]['connected_to'].append('E_' + str(j))
				nr_synapses += 1
			else:
				if ('E_' + str(j)) in netlist['I_' + str(i-Ne)]['connected_to']:
					continue
				netlist['I_' + str(i-Ne)]['connected_to'].append('E_' + str(j))
				nr_synapses += 1

	for j in range(Ni):
		for _ in range(Ci):
			i = int(random.random() * n)
			if i < Ne:
				if ('I_' + str(j)) in netlist['E_' + str(i)]['connected_to']:
					continue
				netlist['E_' + str(i)]['connected_to'].append('I_' + str(j))
				nr_synapses += 1
			else:
				if ('I_' + str(j)) in netlist['I_' + str(i-Ne)]['connected_to']:
					continue
				netlist['I_' + str(i - Ne)]['connected_to'].append('I_' + str(j))
				nr_synapses += 1

	sim_log.message(
		f'Generated netlist for "two population Brunel" testcase:\n'
		f'\tNumber of neurons = {n}\n\tBeta = {beta}\n\tEpsilon = {epsilon}\n\tNr. of synapses = {nr_synapses}\n'
		f'Save netlist in {os.getcwd() + file_name}.json')
	if path:
		with open(path + file_name + '.json', 'w+') as file:
			json.dump(netlist, file)

	return netlist, path + file_name + '.json'


def TwoPopulationNetwork_VogelsAbbott(n = 10000, beta = 0.8, epsilon = 0.01, m = 33, netlist_name = None, path = ''):
	if not netlist_name:
		file_name = f'VogelsAbbott_N{n}_B{beta}_C{epsilon}'
	else:
		file_name = netlist_name

	if os.path.exists(path + file_name + '.json'):
		with open(path + file_name + '.json', 'r') as file:
			netlist = json.load(file)
		sim_log.message(
			f'The simulation folder already contains a netlist file: {file_name}.json\nLoad existing file.')

		if not len(netlist) == n:
			sim_log.fatal_error(
				f'Loaded netlist contains {len(netlist)} neurons, this does not match the specified netlist settings (n := {n}).'
				f'\nNetlist file might be corrupted.'
				f'Rename or remove existing file and restart simulator to force generation a new netlist file.')
		return netlist, path + file_name + '.json'

	Ne = int(beta * n)
	Ni = int(n - Ne)
	netlist = {}
	nr_synapses = 0

	for i in range(Ne):
		netlist['E_' + str(i)] = {
			'FR': 1,
			'connected_to': []}
		for j in range(n):
			p = random.random()
			if p < epsilon and i != j:
				if j < Ne:
					netlist['E_' + str(i)]['connected_to'].append('E_' + str(j))
					nr_synapses += 1
				else:
					netlist['E_' + str(i)]['connected_to'].append('I_' + str(j-Ne))
					nr_synapses += 1

	for i in range(Ni):
		netlist['I_' + str(i)] = {
			'FR': 1,
			'connected_to': []}
		for j in range(n):
			p = random.random()
			if p < epsilon and i != j - Ne:
				if j < Ne:
					netlist['I_' + str(i)]['connected_to'].append('E_' + str(j))
					nr_synapses += 1
				else:
					netlist['I_' + str(i)]['connected_to'].append('I_' + str(j-Ne))
					nr_synapses += 1

	input_nr = []
	while not len(input_nr) == m:
		j = int(random.random() * Ne)
		if j in input_nr:
			continue
		else:
			netlist['S_' + str(len(input_nr))] = {
				'FR': 1,
				'connected_to':  ['E_' + str(j)]}
			input_nr.append(j)

	for neuron in netlist:
		if neuron in netlist[neuron]['connected_to']:
			raise Exception('Autapse detected')

	sim_log.message(
		f'Generated netlist for "two population Vogels Abott" testcase:\n'
		f'\tNumber of neurons = {n}\n\tBeta = {beta}\n\tEpsilon = {epsilon}\n\tNr. of synapses = {nr_synapses}\n'
		f'Save netlist in {os.getcwd() + file_name}.json')
	if path:
		with open(path + file_name + '.json', 'w+') as file:
			json.dump(netlist, file)

	return netlist, path + file_name + '.json'


def CorticalMicrocircuit_PotjansDiesmann(scale_factor = 1, netlist_name = None, FR_defined = False, path = ''):
	# A microcircuit NN model with 78.071 neurons (77.169 LIF Neurons and 902 TC "Neurons")
	# and approx. 291.823.019 synapses (binomial distributed)
	# This method does not exactly match the testcase, the external in-degree is not considered/understood
	if not netlist_name:
		if scale_factor == 1:
			file_name = 'Cortical_Microcircuit'
		else:
			file_name = f'Cortical_Microcircuit_Scale_{str(scale_factor).replace(".","_")}'
	else:
		file_name = netlist_name

	if os.path.exists(path + file_name + '.json'):
		with open(path + file_name + '.json', 'r') as file:
			netlist = json.load(file)
		sim_log.message(
			f'The simulation folder already contains a netlist file: {file_name}.json\nLoad existing file.')

		sim_log.notice(
			f'Loaded netlist contains {len(netlist)} neurons. '
			f'Sould be roughly {round(78071 * scale_factor)} +/- 9 neurons. '
			f'If not, the old netlist might be incomplete or corrupted.'
			f'Rename or remove existing file and restart simulator to force generation a new netlist file.')
		return netlist, path + file_name + '.json'

	C = {
		'L2/3E': {
			'L2/3E': 0.1009,
			'L2/3I': 0.1346,
			'L4E': 0.0077,
			'L4I': 0.0691,
			'L5E': 0.1004,
			'L5I': 0.0548,
			'L6E': 0.0156,
			'L6I': 0.0364
		},
		'L2/3I': {
			'L2/3E': 0.1689,
			'L2/3I': 0.1371,
			'L4E': 0.0059,
			'L4I': 0.0029,
			'L5E': 0.0622,
			'L5I': 0.0269,
			'L6E': 0.0066,
			'L6I': 0.0010
		},
		'L4E': {
			'L2/3E': 0.0437,
			'L2/3I': 0.0316,
			'L4E': 0.0497,
			'L4I': 0.0794,
			'L5E': 0.0505,
			'L5I': 0.0257,
			'L6E': 0.0211,
			'L6I': 0.0034
		},
		'L4I': {
			'L2/3E': 0.0818,
			'L2/3I': 0.0515,
			'L4E': 0.1350,
			'L4I': 0.1597,
			'L5E': 0.0057,
			'L5I': 0.0022,
			'L6E': 0.0166,
			'L6I': 0.0005
		},
		'L5E': {
			'L2/3E': 0.0323,
			'L2/3I': 0.0755,
			'L4E': 0.0067,
			'L4I': 0.0033,
			'L5E': 0.0831,
			'L5I': 0.0600,
			'L6E': 0.0572,
			'L6I': 0.0277
		},
		'L5I': {
			'L2/3E': 0.0,
			'L2/3I': 0.0,
			'L4E': 0.0003,
			'L4I': 0.0,
			'L5E': 0.3726,
			'L5I': 0.3158,
			'L6E': 0.0197,
			'L6I': 0.0080
		},
		'L6E': {
			'L2/3E': 0.0076,
			'L2/3I': 0.0042,
			'L4E': 0.0453,
			'L4I': 0.1057,
			'L5E': 0.0204,
			'L5I': 0.0086,
			'L6E': 0.0396,
			'L6I': 0.0658
		},
		'L6I': {
			'L2/3E': 0.0,
			'L2/3I': 0.0,
			'L4E': 0.0,
			'L4I': 0.0,
			'L5E': 0.0,
			'L5I': 0.0,
			'L6E': 0.2252,
			'L6I': 0.1443
		},
		'TC': {
			'L2/3E': 0.0,
			'L2/3I': 0.0,
			'L4E': 0.0983,
			'L4I': 0.0619,
			'L5E': 0.0,
			'L5I': 0.0,
			'L6E': 0.0512,
			'L6I': 0.196
		}
	}

	N = {
		'L2/3E': int(20683 * scale_factor),
		'L2/3I': int(5834 * scale_factor),
		'L4E': int(21915 * scale_factor),
		'L4I': int(5479 * scale_factor),
		'L5E': int(4850 * scale_factor),
		'L5I': int(1065 * scale_factor),
		'L6E': int(14395 * scale_factor),
		'L6I': int(2948 * scale_factor),
		'TC': int(902 * scale_factor)
	}

	FR = {
		'L2/3E': 0.84912959381,
		'L2/3I': 3.59793814432,
		'L4E': 3.89041095890,
		'L4I': 7.02930402930,
		'L5E': 8.42975206611,
		'L5I': 9.24528301886,
		'L6E': 1.15159944367,
		'L6I': 8.59863945578,
		'TC': 1
	}

	netlist = {}
	nr_synapses = 0
	for key in N.keys():
		for neuron in range(N[key]):
			neuron_ID = f'{key}_{neuron}'
			connectionlist = []

			for Ckey in C[key].keys():
				for synapse in range(N[Ckey]):
					if random.random() <= C[key][Ckey]:
						connectionlist.append(f'{Ckey}_{synapse}')
						nr_synapses += 1

			if FR_defined:
				netlist[neuron_ID] = {
					'FR': FR[key],
					'connected_to': connectionlist
				}
			else:
				netlist[neuron_ID] = {
					'FR': 1,
					'connected_to': connectionlist
				}

	sim_log.message(
		f'Generated netlist for Potjans Diesmann Cortical Microcircuit model:\n'
		f'\tScale Factor = {scale_factor}\n\tNumber of neurons = {len(netlist)}\n\tNr. of synapses = {nr_synapses}\n'
		f'Save netlist in {os.getcwd() + file_name}.json')
	if path:
		with open(path + file_name + '.json', 'w+') as file:
			json.dump(netlist, file)

	return netlist, path + file_name + '.json'


def CorticalMicrocircuit_Benchmark(N=0.5, K=0.2, netlist_name = None, path = ''):
	# A microcircuit NN model which can be scaled both in number of neurons (N) and number of synapses per neuron (K)
	# Testcase used to validate the simulator against hardware
	# Opposed to original CM-Model, this testcase does not include the TC population.
	# However, it does include SpikeSourcePopulations (SRC) and DelayExtensionPopulations (DE),
	# one for every regular Integrate-and-Fire Population (IF).
	correction_factor = K/N

	if not netlist_name:
		file_name = f'Cortical_Microcircuit_N{int(N*100):02d}-K{int(K*100):02d}'
	else:
		file_name = netlist_name

	if os.path.exists(path + file_name + '.json'):
		with open(path + file_name + '.json', 'r') as file:
			netlist = json.load(file)
		sim_log.message(
			f'The simulation folder already contains a netlist file: {file_name}.json\nLoad existing file.')

		sim_log.notice(
			f'Loaded netlist contains {len(netlist)} neurons. '
			f'Sould be roughly {round(78071 * N)} +/- 24 neurons. '
			f'If not, the old netlist might be incomplete or corrupted.'
			f'Rename or remove existing file and restart simulator to force generation a new netlist file.')
		return netlist, path + file_name + '.json'

	C = {
		'L2/3E': {
			'L2/3E': 0.1009,
			'L2/3I': 0.1346,
			'L4E': 0.0077,
			'L4I': 0.0691,
			'L5E': 0.1004,
			'L5I': 0.0548,
			'L6E': 0.0156,
			'L6I': 0.0364
		},
		'L2/3I': {
			'L2/3E': 0.1689,
			'L2/3I': 0.1371,
			'L4E': 0.0059,
			'L4I': 0.0029,
			'L5E': 0.0622,
			'L5I': 0.0269,
			'L6E': 0.0066,
			'L6I': 0.0010
		},
		'L4E': {
			'L2/3E': 0.0437,
			'L2/3I': 0.0316,
			'L4E': 0.0497,
			'L4I': 0.0794,
			'L5E': 0.0505,
			'L5I': 0.0257,
			'L6E': 0.0211,
			'L6I': 0.0034
		},
		'L4I': {
			'L2/3E': 0.0818,
			'L2/3I': 0.0515,
			'L4E': 0.1350,
			'L4I': 0.1597,
			'L5E': 0.0057,
			'L5I': 0.0022,
			'L6E': 0.0166,
			'L6I': 0.0005
		},
		'L5E': {
			'L2/3E': 0.0323,
			'L2/3I': 0.0755,
			'L4E': 0.0067,
			'L4I': 0.0033,
			'L5E': 0.0831,
			'L5I': 0.0600,
			'L6E': 0.0572,
			'L6I': 0.0277
		},
		'L5I': {
			'L2/3E': 0.0,
			'L2/3I': 0.0,
			'L4E': 0.0003,
			'L4I': 0.0,
			'L5E': 0.3726,
			'L5I': 0.3158,
			'L6E': 0.0197,
			'L6I': 0.0080
		},
		'L6E': {
			'L2/3E': 0.0076,
			'L2/3I': 0.0042,
			'L4E': 0.0453,
			'L4I': 0.1057,
			'L5E': 0.0204,
			'L5I': 0.0086,
			'L6E': 0.0396,
			'L6I': 0.0658
		},
		'L6I': {
			'L2/3E': 0.0,
			'L2/3I': 0.0,
			'L4E': 0.0,
			'L4I': 0.0,
			'L5E': 0.0,
			'L5I': 0.0,
			'L6E': 0.2252,
			'L6I': 0.1443
		}
	}

	NPops = {
		'L2/3E': int(20683 * N),
		'L2/3I': int(5834 * N),
		'L4E': int(21915 * N),
		'L4I': int(5479 * N),
		'L5E': int(4850 * N),
		'L5I': int(1065 * N),
		'L6E': int(14395 * N),
		'L6I': int(2948 * N),
		'SRC_L2/3E': int(20683 * N),
		'SRC_L2/3I': int(5834 * N),
		'SRC_L4E': int(21915 * N),
		'SRC_L4I': int(5479 * N),
		'SRC_L5E': int(4850 * N),
		'SRC_L5I': int(1065 * N),
		'SRC_L6E': int(14395 * N),
		'SRC_L6I': int(2948 * N),
		'DE_L2/3E': int(20683 * N),
		'DE_L2/3I': int(5834 * N),
		'DE_L4E': int(21915 * N),
		'DE_L4I': int(5479 * N),
		'DE_L5E': int(4850 * N),
		'DE_L5I': int(1065 * N),
		'DE_L6E': int(14395 * N),
		'DE_L6I': int(2948 * N)
	}

	FRPops = {
		'L2/3E': 0.84912959381,
		'L2/3I': 3.59793814432,
		'L4E': 3.89041095890,
		'L4I': 7.02930402930,
		'L5E': 8.42975206611,
		'L5I': 9.24528301886,
		'L6E': 1.15159944367,
		'L6I': 8.59863945578,
		'SRC_L2/3E': 2560.78336557,
		'SRC_L2/3I': 2422.62199312,
		'SRC_L4E': 3362.28036529,
		'SRC_L4I': 3041.53113553,
		'SRC_L5E': 3195.96694214,
		'SRC_L5I': 3048.67924528,
		'SRC_L6E': 4635.81363004,
		'SRC_L6I': 3353.82993197,
		'DE_L2/3E': 0.84912959381,
		'DE_L2/3I': 3.59793814432,
		'DE_L4E': 3.89041095890,
		'DE_L4I': 7.02930402930,
		'DE_L5E': 8.42975206611,
		'DE_L5I': 9.24528301886,
		'DE_L6E': 1.15159944367,
		'DE_L6I': 8.59863945578
	}

	netlist = {}
	nr_synapses = 0
	for key in ['L2/3E', 'L2/3I', 'L4E', 'L4I', 'L5E', 'L5I', 'L6E', 'L6I']:
		for neuron in range(NPops[key]):
			neuron_ID = f'{key}_{neuron}'
			connectionlist = []

			for Ckey in C[key].keys():
				for synapse in range(NPops[Ckey]):
					if random.random() <= C[key][Ckey] * correction_factor:
						connectionlist.append(f'{Ckey}_{synapse}')
						nr_synapses += 1

				# Ensure at least one synapse per population if the probability > 0
				if C[key][Ckey] and (not connectionlist or not connectionlist[-1].startswith(Ckey)):
					synapse = random.randint(0, NPops[Ckey] - 1)
					connectionlist.append(f'{Ckey}_{synapse}')

			Delay_neuron = f'DE_{neuron_ID}'
			netlist[Delay_neuron] = {
				'FR': FRPops[f'DE_{key}'],
				'connected_to': connectionlist
			}

			connectionlist.append(f'DE_{key}_{neuron}')
			netlist[neuron_ID] = {
				'FR': FRPops[key],
				'connected_to': connectionlist
			}

	for key in ['SRC_L2/3E', 'SRC_L2/3I', 'SRC_L4E', 'SRC_L4I', 'SRC_L5E', 'SRC_L5I', 'SRC_L6E', 'SRC_L6I']:
		for neuron in range(NPops[key]):
			neuron_ID = f'{key}_{neuron}'
			target_neuron = f'{key[4:]}_{neuron}'
			netlist[neuron_ID] = {
				'FR': FRPops[key],
				'connected_to': [target_neuron]
			}

	sim_log.message(
		f'Generated netlist for a scaled N{int(N*100):02d}-K{int(K*100):02d} Cortical Microcircuit model:\n'
		f'\n\tNumber of neurons = {len(netlist) / 3} (* 3 due to SRC and DE pops)\n\tNr. of synapses = {nr_synapses}\n'
		f'Save netlist in {os.getcwd() + path + file_name}.json')
	if path:
		with open(path + file_name + '.json', 'w+') as file:
			json.dump(netlist, file)

	return netlist, path + file_name + '.json'


########################################################################################################################
# Analytical method
########################################################################################################################
# Can be excecuted manually from the terminal to calculate the number of neurons and synapses in a specific netlist file
# Not used by the simulator itself
def Analyse_netlist(file_name):
	if os.path.exists(file_name + '.json'):
		with open(file_name + '.json', 'r') as file:
			netlist = json.load(file)

		SpNeuron = []
		tot_syn = 0
		for neuron, data in netlist.items():
			synapses = data['connected_to']
			if neuron.startswith("L"):
				SpNeuron.append(len(synapses))
				tot_syn += len(synapses)

		print(f'Total number of Synapses in model: {tot_syn}')
		print(f'Average number of Synapses per neuron: {sum(SpNeuron)/len(SpNeuron)}')

