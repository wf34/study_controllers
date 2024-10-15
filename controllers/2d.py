#!/usr/bin/env python3

from tap import Tap

from pydrake.all import (
    DiagramBuilder,
    #DiscreteContactApproximation,
    AddMultibodyPlantSceneGraph,
    JacobianWrtVariable,
    ProcessModelDirectives,
    ModelDirectives,
    LeafSystem,
    RollPitchYaw,
    Simulator,
    StartMeshcat,
    VectorLogSink,
    ApplyMultibodyPlantConfig,
    RobotDiagramBuilder,
    Parser,
)

from resource_loader import get_resource, get_resource_path, LoadScenario

TIME_STEP=0.001  # faster


class TwoDArgs(Tap):
    target_browser_for_replay: str = 'chromium' #'xdg-open'


def simulate_2d(args: TwoDArgs):
    meshcat = StartMeshcat()
    meshcat.Set2dRenderMode(xmin=-0.25, xmax=1.5, ymin=-0.1, ymax=1.3)
    builder = DiagramBuilder()

    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=TIME_STEP)

    robot_builder = RobotDiagramBuilder(time_step=TIME_STEP)
    sim_plant = robot_builder.plant()
    scene_graph = robot_builder.scene_graph()
    scenario = LoadScenario(filename=get_resource_path('planar_manipulation_station.scenario.yaml', arg_provides_ext=True))
    ApplyMultibodyPlantConfig(scenario.plant_config, sim_plant)

    parser = Parser(sim_plant)

    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        parser=parser,
    )

    # j 2, 4, 6 ?

if '__main__' == __name__:
    simulate_2d(TwoDArgs().parse_args())
