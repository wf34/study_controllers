#!/usr/bin/env python3

import os
import json
import typing

import numpy as np
from tap import Tap
from matplotlib import pyplot as plt

from state_monitor import Datum

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
)

class PlotterArgs(Tap):
    input_files: typing.List[str]
    labels: typing.List[str] = []
    mode: typing.Literal['pe', 'f', 'p']
    sparse_load: bool = False

    def process_args(self):
        if len(list(map(os.path.isfile, self.input_files))) != len(self.input_files):
            raise Exception(f'non-existing files were used')
        if len(self.labels) != 0 and len(self.input_files) != len(self.labels):
            raise Exception('mismatch in labels: {} {}'.format(len(self.input_files), len(self.labels)))
        if len(self.labels) == 0:
            self.labels = list(map(lambda x: os.path.basename(x)[:-5], self.input_files))


def load_datums(input_file: str, do_sparse_load: bool = False):
    datums = []
    with open(input_file, 'r') as the_file:
        file_contents = the_file.read()
        json_struct = json.loads(file_contents)
        datums_cnt = 0
        for datum_raw in json_struct:
            if not do_sparse_load or 0 == datums_cnt % 10:
                datums.append(Datum(**datum_raw))
            datums_cnt += 1
    return datums


def plot(args: PlotterArgs):
    per_file_datums = list(map(lambda x: load_datums(x, args.sparse_load), args.input_files))
    
    trajectories_amount = len(per_file_datums)
    print(trajectories_amount, len(args.labels), '/////')

    fig = plt.figure()
    ax = fig.subplots(nrows=1, ncols=1)
    ax_secondary = ax.twinx()
    ax_secondary.yaxis.set_ticks(np.arange(0, 4))

    NUM_COLORS = trajectories_amount
    cm = plt.get_cmap('flag')
    ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    is_first = True
    lines = []
    for datums, label in zip(per_file_datums, args.labels):
        t = []
        tn = []
        pitches = []
        for d in datums:
            t.append(d.time)
            tn.append(d.turn)
            pitches.append(d.valve_pitch_angle)

        if is_first:
            l1 = ax_secondary.plot(t, tn, label='turn count', color='k')
            is_first = False
            lines.extend(l1)

        if 'pe' == args.mode:
            pass
        elif 'f' == args.mode:
            pass
        elif 'p' == args.mode:
            l2 = ax.plot(t, pitches, label=f'{label}/pitch')
            lines.extend(l2)
        else:
            raise Exception('unreachable')

    for offset in range(3):
        real_offset = offset * 22
        ax.axvline(real_offset + 6., color='red', linestyle='-.', alpha=1)
        ax.axvline(real_offset + 16., color='green', linestyle='-.', alpha=1)

    ax.legend(lines, [l.get_label() for l in lines])
    plt.show()


if __name__ == '__main__':
    args = PlotterArgs().parse_args()
    plot(args)
