#!/usr/bin/env python3

import os
import json

from tap import Tap
from matplotlib import pyplot as plt

from state_monitor import Datum

class PlotterArgs(Tap):
    input_file: str

    def process_args(self):
        if not os.path.isfile(self.input_file):
            raise Exception(f'empty input: {self.input_file}')


def plot_forces(args: PlotterArgs):
    datums = []
    with open(args.input_file, 'r') as the_file:
        file_contents = the_file.read()
        file_contents = file_contents[:-1] + ']'
        json_struct = json.loads(file_contents)
        for datum_raw in json_struct:
            datums.append(Datum(**datum_raw))
    print('datum: ', len(datums))

    t = []
    fx = []
    fy = []
    theta = []

    for d in datums:
        t.append(d.time)
        fx.append(d.reaction_forces.x)
        fy.append(d.reaction_forces.y)
        theta.append(d.reaction_forces.z)

    plt.plot(t, fx, color='red', linewidth=4)
    plt.plot(t, fy, color='green', linewidth=2)
    plt.plot(t, theta, color='blue')

    plt.show()


if __name__ == '__main__':
    args = PlotterArgs().parse_args()
    plot_forces(args)
