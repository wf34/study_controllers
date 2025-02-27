#!/usr/bin/env python3

import os
import operator
import json
import typing
import math

import numpy as np
from tap import Tap
#import matplotlib
#matplotlib.use('GTK3Agg')
from matplotlib import pyplot as plt

from state_monitor import Datum
from planning import TurnStage

from pydrake.all import (
    RigidTransform,
    RotationMatrix,
)

class PlotterArgs(Tap):
    input_files: typing.List[str]
    labels: typing.List[str] = []
    mode: typing.Literal['pe', 'f', 'p']
    sparse_load: bool = False
    destination: typing.Optional[str] = None

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
            if not do_sparse_load or 0 == datums_cnt % 400:
                datums.append(Datum(**datum_raw))
            datums_cnt += 1
    return datums


def plot(args: PlotterArgs):
    per_file_datums = list(map(lambda x: load_datums(x, args.sparse_load), args.input_files))
    
    trajectories_amount = len(per_file_datums)

    fig = plt.figure()
    ax = fig.subplots(nrows=1, ncols=1)
    ax_secondary = ax.twinx()
    ax_secondary.yaxis.set_ticks(np.arange(0, 4))

    #NUM_COLORS = trajectories_amount
    #cm = plt.get_cmap('flag')
    #ax.set_prop_cycle(color=[cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
    main_colors = ['r', 'k', 'green']
    assert trajectories_amount <= len(main_colors)
    lines = []
    labels = []
    starting_pitch = None
    pitch_for_vis = lambda x: np.degrees(x - starting_pitch)
    first_trajectory = True

    for datums, label, mc in zip(per_file_datums, args.labels, main_colors):
        t = []
        t_by_stage = {}
        tn_by_stage = {}
        pitches = []
        Ms = []
        t_des = []
        t_mes = []
        q_des = []
        q_mes = []

        for d in datums:
            t.append(d.time)
            if d.force_sensed is not None:
                Ms.append(d.force_sensed.x)
            else:
                Ms.append(None)

            assert d.Xee_desired_W is not None and d.Xee_observed_W is not None
            t_des.append(d.Xee_desired_W.translation.to_np())
            t_mes.append(d.Xee_observed_W.translation.to_np())
            q_des.append(d.Xee_desired_W.rotation.to_drake_q())
            q_mes.append(d.Xee_observed_W.rotation.to_drake_q())

            t_by_stage.setdefault(d.stage, []).append(d.time)
            tn_by_stage.setdefault(d.stage, []).append(d.turn + 1 if d.stage != TurnStage.RETRACT else d.turn)
            if starting_pitch is None:
                pitches.append(0.)
                starting_pitch = d.valve_pitch_angle
            else:
                pitches.append(pitch_for_vis(d.valve_pitch_angle))


        if 'pe' == args.mode:
            if first_trajectory:
                for stage, c in zip([TurnStage.APPROACH, TurnStage.SCREW, TurnStage.RETRACT],
                                    ['orange', 'yellow', 'cyan']):
                                    #[(1, .49, 0.), (1, 1, 0), (0., .54, .54)]):
                    l_sec = ax_secondary.scatter(t_by_stage[stage], tn_by_stage[stage], color=c, s=2**8, alpha=0.33, edgecolors='none')
                    lines.append(l_sec)
                    labels.append(f'{stage.value} at turn #')

                first_trajectory = False

            t_des = np.array(t_des)
            t_mes = np.array(t_mes)
            err = np.square(t_des - t_mes)
            err = np.sqrt(np.sum(err, axis=1))

            q_err = []
            for qd, qm in zip(q_des, q_mes):
                q_rez = qd.inverse().multiply(qm)
                qnor = np.linalg.norm(q_rez.wxyz())
                q_err.append(q_rez.w() / qnor)
                assert -1. <= q_err[-1] <= 1.

            ang_err = np.degrees(np.abs(np.arccos(q_err))) * 2.

            lf = ax.plot(t, err, label=label, color=mc)
            lines.extend(lf)
            labels.append(lf[0].get_label())

        elif 'f' == args.mode:
            if True:
                inds = []
                ind_ends = []
                turn_id_to_screw_start_time = lambda i:  22. * i + 6.
                turn_id_to_screw_end_time = lambda i: turn_id_to_screw_start_time(i) + 10.
                for x in range(3):
                    cut_val = turn_id_to_screw_start_time(x)
                    cut_val_end = turn_id_to_screw_end_time(x)
                    find_index = lambda cut_value : next(map(operator.itemgetter(1), filter(lambda x: math.fabs(x[0] - cut_value) < 1.e-3, zip(t, range(len(t))))))
                    inds.append(find_index(cut_val))
                    ind_ends.append(find_index(cut_val_end))

                times_ = []
                moments = []
                min_len = None
                for i in range(3):
                    times_.append(t[inds[i]: ind_ends[i]])
                    moments.append(Ms[inds[i]: ind_ends[i]])
                    cur_len = len(moments[-1])
                    if min_len is None or cur_len < min_len:
                        min_len = cur_len
                for i in range(3):
                    times_[i] = times_[i][1:min_len]
                    moments[i] = moments[i][1:min_len]

                moments = np.mean(moments, axis=0)
                print(moments.shape)
                lf = ax.plot(times_[0], moments, label=label, color=mc)
                lines.extend(lf)
                labels.append(lf[0].get_label())
                if first_trajectory:
                    first_trajectory = False
                    lh = ax.axhline(y=-.2, color=main_colors[-1], linestyle='-.', label='target_moment')
                    lines.append(lh)
                    labels.append(lh.get_label())
            else:
                lf = ax.plot(t, Ms, label=label, color=mc)
                lines.extend(lf)
                labels.append(lf[0].get_label())

        elif 'p' == args.mode:
            if first_trajectory:
                for stage, c in zip([TurnStage.APPROACH, TurnStage.SCREW, TurnStage.RETRACT],
                                    ['orange', 'yellow', 'cyan']):
                                    #[(1, .49, 0.), (1, 1, 0), (0., .54, .54)]):
                    l_sec = ax_secondary.scatter(t_by_stage[stage], tn_by_stage[stage], color=c, s=2**8, alpha=0.33, edgecolors='none')
                    lines.append(l_sec)
                    labels.append(f'{stage.value} at turn #')

                first_trajectory = False
            l2 = ax.plot(t, pitches, label=f'{label}', color=mc)
            lines.extend(l2)
            labels.append(lines[-1].get_label())
        else:
            raise Exception('unreachable')

    if args.mode in ('p', 'pe'):
        ax.set_zorder(ax_secondary.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax_secondary.set_ylabel('Turn #')
        if args.mode == 'p':
            ax.set_ylabel('Nut Turning Progress, degrees')
        elif args.mode == 'pe':
            #ax.set_ylim(0., 30.)
            #ax.set_ylabel('Planned Trajectory Tracking Error, degrees')
            ax.set_ylabel('Planned Trajectory Tracking Error, meters')
    elif args.mode == 'f':
        ax.set_ylabel('Nut-Gripper Average Contact Moment, N•m')
    ax.legend(lines, labels, loc = "upper left")
    ax.set_xlabel('Simulation Time, seconds')
    plt.tight_layout()

    if args.destination is not None:
        assert args.destination.endswith('.eps')
        fig.savefig(args.destination, format='eps')
    else:
        plt.show()



if __name__ == '__main__':
    args = PlotterArgs().parse_args()
    plot(args)
