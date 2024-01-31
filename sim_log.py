"""
sim_log.py

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

log_file = 'unnamed_file.log'
notice_counter = 0
warning_counter = 0
error_counter = 0


class CastingError(Exception):
	pass


class RoutingError(Exception):
	pass


class MappingError(Exception):
	pass


class SimError(Exception):
	pass


def create_log(run_name, write_config):
	global log_file
	if os.path.isfile(run_name + '.log'):
		log_file = run_name + ' ' + time.ctime() + '.log'
	else:
		log_file = run_name + '.log'
	with open(log_file, 'w') as file:
		file.write(
			f'================================================================================================\n'
			f'Simulation: {run_name} started at {time.ctime()}\n'
			f'================================================================================================\n'
			f'Settings for current run:\n')

	with open(log_file, 'a') as file:
		write_config.write(file)

	with open(run_name + '.ini', 'w') as cfgfile:
		write_config.write(cfgfile)

	print(
		f'================================================================================================\n'
		f'Simulation: {run_name} started at {time.ctime()}\n'
		f'================================================================================================\n')


def end_log(run_time):
	with open(log_file, 'a') as file:
		file.write(
			f'\n\nNumber of Notifications: {notice_counter}\n'
			f'Number of WARNINGS: {warning_counter}\n'
			f'Number of ERRORS: {error_counter}\n\n'
			f'================================================================================================\n'
			f'Simulation ended at {time.ctime()}\n'
			f'Run time: {run_time}\n'
			f'================================================================================================\n')
	print(
		f'\n\nNumber of Notifications: {notice_counter}\n'
		f'Number of WARNINGS: {warning_counter}\n'
		f'Number of ERRORS: {error_counter}\n\n'
		f'\n================================================================================================\n'
		f'Simulation ended at {time.ctime()}\n'
		f'Run time: {run_time}\n'
		f'================================================================================================\n')


def message(log_message):
	with open(log_file, 'a') as file:
		file.write(f'{log_message}\n')


def notice(log_message):
	global notice_counter
	notice_counter += 1
	with open(log_file, 'a') as file:
		file.write(
			f'NOTICE!\n'
			f'{log_message}\n')
	print(
		f'NOTICE!\n'
		f'{log_message}\n')


def warning(log_message):
	global warning_counter
	warning_counter += 1
	with open(log_file, 'a') as file:
		file.write(
			f'WARNING!\n'
			f'{log_message}\n')
	print(
		f'WARNING!\n'
		f'{log_message}\n')


def error(log_message):
	global error_counter
	error_counter += 1
	with open(log_file, 'a') as file:
		file.write(
			f'ERROR!\n'
			f'{log_message}\n\n')
	print(
		f'ERROR\n'
		f'{log_message}\n\n')


def fatal_error(log_message):
	with open(log_file, 'a') as file:
		file.write(
			f"\n\nFATAL ERROR\n"
			f"{log_message}\n\n"
			f"Number of Notifications: {notice_counter}\n"
			f"Number of WARNINGS: {warning_counter}\n"
			f"Number of ERRORS: {error_counter}\n"
			f"Simulation terminated at: {time.ctime()}")
	print(
		f"FATAL ERROR\n"
		f"{log_message}\n\n"
		f"Number of WARNINGS: {warning_counter}\n"
		f"Number of ERRORS: {error_counter}\n\n"
		f"Simulation terminated at: {time.ctime()}")

	sys.exit()
