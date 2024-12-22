#!/usr/bin/env python3

import os
import json
import typing

from tap import Tap
from matplotlib import pyplot as plt

from state_monitor import Datum

class PlotterArgs(Tap):
    input_files: typing.List[str]
    labels: typing.List[str] = []
    plot_what: typing.Literal['pe', 'f']

    def process_args(self):
        if len(list(map(os.path.isfile, self.input_files))) != len(self.input_files):
            raise Exception(f'non-existing files were used')
        if len(self.labels) != 0 and len(self.input_files) != len(self.labels):
            raise Exception('mismatch in labels: {} {}'.format(len(self.input_files), len(self.labels)))
        if len(self.labels) == 0:
            self.labels = list(map(str, range(len(self.input_files))))


def load_datums(input_file: str):
    datums = []
    with open(input_file, 'r') as the_file:
        file_contents = the_file.read()
        file_contents = file_contents[:-1] + ']'
        json_struct = json.loads(file_contents)
        for datum_raw in json_struct:
            datums.append(Datum(**datum_raw))
    return datums

def plot(args: PlotterArgs):
    per_file_datums = list(map(load_datums, args.input_files))

    NUM_COLORS = len(args.input_files)

    fig = plt.figure()
    ax = fig.subplots(nrows=1, ncols=2)

    cm = plt.get_cmap('flag')#gist_rainbow')
    for i in range(2):
        ax[i].set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])

    label_prefix = args.plot_what
    for datums, label_, w in zip(per_file_datums, args.labels, [2, 1]):
        t = []
        pe_x = []
        pe_z = []

        for d in datums:
            t.append(d.time)
            if args.plot_what == 'pe':
                pe_x.append(d.pe_s.x)
                pe_z.append(d.pe_s.z +0.11826275 if label_ == 'hybrid' else 0.)
            elif args.plot_what == 'f':
                pe_x.append(d.f_s.x)
                pe_z.append(d.f_s.z)

        ax[0].plot(t, pe_x, linewidth=w, label=label_ + f'/{label_prefix}_x')
        ax[1].plot(t, pe_z, linewidth=w, label=label_ + f'/{label_prefix}_z')

    for i in range(2):
        ax[i].legend()
        ax[i].axhline(0., color='k', linestyle='-.', alpha=1)

    plt.show()


if __name__ == '__main__':
    args = PlotterArgs().parse_args()
    plot(args)
