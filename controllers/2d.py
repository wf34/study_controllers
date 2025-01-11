#!/usr/bin/env python3

import time
import os
import typing
import dataclasses as dc

import numpy as np
from tap import Tap
import pydot

from pydrake.all import (
    RollPitchYaw,
    Sphere,
    Rgba,
    Adder,
    ApplyLcmBusConfig,
    ContactVisualizer,
    ContactVisualizerParams,
    DrakeLcmParams,
    Diagram,
    DiagramBuilder,
    DiscreteContactApproximation,
    ApplyMultibodyPlantConfig,
    ApplyVisualizationConfig,
    FlattenModelDirectives,
    ProcessModelDirectives,
    ModelDirective,
    ModelDirectives,
    ModelInstanceIndex,
    ModelInstanceInfo,
    MeshcatVisualizer,
    LeafSystem,
    MakeMultibodyStateToWsgStateSystem,
    Simulator,
    Meshcat,
    StartMeshcat,
    MultibodyPlant,
    VectorLogSink,
    RobotDiagramBuilder,
    Parser,
    IiwaDriver,
    SchunkWsgDriver,
    SchunkWsgPositionController,
    WeldJoint,
    RigidTransform,
    RotationMatrix,
    SharedPointerSystem,
    ParseIiwaControlMode,
    SimIiwaDriver,
    TrajectorySource,
)

from directives_tree import DirectivesTree
from resource_loader import get_resource_path, LoadScenario, Scenario
from catalog import TrajFollowingJointStiffnessController, HybridCartesianController, ForceSensor
from opt_trajectory import optimize_target_trajectory, make_cartesian_trajectory, make_wsg_trajectory, make_dummy_wsg_trajectory
from state_monitor import StateMonitor
from visualization_tools import AddMeshcatTriad

TIME_STEP=0.001  # faster
INITIAL_IIWA_COORDS = np.array([np.pi / 8., -np.pi / 4., -np.pi / 2.])
SHELF_LENGTH = .5
SHELF_THICKNESS = .075

def minimalist_traj_vis(traj):
    step = 1. / 33
    for ts in np.arange(0, traj.end_time(), step):
        x = traj.value(ts)
        print(' '.join(list(map(lambda y: f'{y:.2f}', [ts] + x.ravel().tolist()))))


def rich_trajectory_vis(trajectory, global_context, plant, visualizer, meshcat):
    step = 1. / 33
    end_time = step if trajectory is None else trajectory.end_time()

    plant_context = plant.GetMyContextFromRoot(global_context)
    visualizer_context = visualizer.GetMyContextFromRoot(global_context)
    visualizer.StartRecording(False)

    for ts in np.arange(0, end_time, step):
        global_context.SetTime(ts)
        if trajectory is not None:
            plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa"), trajectory.value(ts))
        if ts == 0:
            print('but in vis:', plant.GetPositions(plant_context))

        visualizer.ForcedPublish(visualizer_context)

    X_WGcurrent = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName('body'))
    AddMeshcatTriad(meshcat, 'gripper-at-the-end', X_PT=X_WGcurrent)

    visualizer.StopRecording()
    visualizer.PublishRecording()


def _FreezeChildren(
    plant: MultibodyPlant,
    children_to_freeze: typing.List[str],
) -> None:
    """
    Freeze the joints of the given children in the plant.

    Freezing really means removing a joint belonging to any of the children,
    and replacing it with a weld joint.
    """
    if len(children_to_freeze) == 0:
        return

    # Enumerate joints that need to be frozen i.e. removed and replaced by weld
    # joints. These are joints of child instances that are not already welds.
    joints_to_freeze: typing.Set[Joint] = set()
    for child_instance_name in children_to_freeze:
        child_instance = plant.GetModelInstanceByName(child_instance_name)
        for joint_index in plant.GetJointIndices(child_instance):
            joint = plant.get_joint(joint_index)
            if joint.type_name() != "weld":
                joints_to_freeze.add(joint)

    # Before removing joints, we need to remove associated actuators.
    for actuator_index in plant.GetJointActuatorIndices():
        actuator = plant.get_joint_actuator(actuator_index)
        if actuator.joint() in joints_to_freeze:
            plant.RemoveJointActuator(actuator)

    # Remove non-weld joints and replace them with weld joints.
    for joint in joints_to_freeze:
        weld = WeldJoint(
            joint.name(),
            joint.frame_on_parent(),
            joint.frame_on_child(),
            RigidTransform(),
        )
        plant.RemoveJoint(joint)
        plant.AddJoint(weld)


@dc.dataclass
class JointPidControllerGains:
    """Defines the Proportional-Integral-Derivative gains for a single joint.

    Args:
        kp: The proportional gain.
        ki: The integral gain.
        kd: The derivative gain.
    """

    kp: float = 100  # Position gain
    ki: float = 1  # Integral gain
    kd: float = 20  # Velocity gain


@dc.dataclass
class InverseDynamicsDriver:
    """A simulation-only driver that adds the InverseDynamicsController to the
    station and exports the output ports. Multiple model instances can be driven with a
    single controller using `instance_name1+instance_name2` as the key; the output ports
    will be named similarly."""

    # Must have one element for every (named) actuator in the model_instance.
    gains: typing.Mapping[str, JointPidControllerGains] = dc.field(default_factory=dict)


@dc.dataclass
class JointPdControllerGains:
    """Defines the Proportional-Derivative gains for a single joint.

    Args:
        kp: The proportional gain.
        kd: The derivative gain.
    """

    kp: float = 0  # Position gain
    kd: float = 0  # Velocity gain


@dc.dataclass
class JointStiffnessDriver:
    """A simulation-only driver that sets up MultibodyPlant to act as if it is
    being controlled with a JointStiffnessController. The MultibodyPlant must
    be using SAP as the (discrete-time) contact solver.

    Args:
        gains: A mapping of {actuator_name: JointPdControllerGains} for each
            actuator that should be controlled.
        hand_model_name: If set, then the gravity compensation will be turned
            off for this model instance (e.g. for a hand).
    """

    # Must have one element for every (named) actuator in the model_instance.
    gains: typing.Mapping[str, JointPdControllerGains] = dc.field(default_factory=dict)

    hand_model_name: str = ""





def _PopulatePlantOrDiagram(
    plant: MultibodyPlant,
    parser: Parser,
    scenario: Scenario,
    model_instance_names: typing.List[str],
    add_frozen_child_instances: bool = True,
    package_xmls: typing.List[str] = [],
    parser_preload_callback: typing.Callable[[Parser], None] = None,
    parser_prefinalize_callback: typing.Callable[[Parser], None] = None,
) -> None:
    """See MakeMultibodyPlant and MakeRobotDiagram for details."""
    ApplyMultibodyPlantConfig(scenario.plant_config, plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    ConfigureParser(parser)
    if parser_preload_callback:
        parser_preload_callback(parser)

    # Make the plant for the iiwa controller to use.
    flattened_directives = FlattenModelDirectives(
        ModelDirectives(directives=scenario.directives), parser.package_map()
    ).directives

    tree = DirectivesTree(flattened_directives)
    directives = tree.GetWeldToWorldDirectives(model_instance_names)
    children_to_freeze = set()

    if add_frozen_child_instances:
        children_to_freeze, additional_directives = (
            tree.GetWeldedDescendantsAndDirectives(model_instance_names)
        )
        directives.extend(additional_directives)

    ProcessModelDirectives(
        directives=ModelDirectives(directives=directives),
        parser=parser,
    )

    _FreezeChildren(plant, children_to_freeze)

    if parser_prefinalize_callback:
        parser_prefinalize_callback(parser)

    plant.Finalize()


def MakeMultibodyPlant(
    scenario: Scenario,
    *,
    model_instance_names: typing.List[str] = None,
    add_frozen_child_instances: bool = False,
    package_xmls: typing.List[str] = [],
    parser_preload_callback: typing.Callable[[Parser], None] = None,
    parser_prefinalize_callback: typing.Callable[[Parser], None] = None,
) -> MultibodyPlant:
    """Use a scenario to create a MultibodyPlant. This is intended, e.g., to facilitate
    easily building subsets of a scenario, for instance, to make a plant for a
    controller.

    Args:
        scenario: A Scenario structure, populated using the `load_scenario`
            method.

        model_instance_names: If specified, then only the named model instances
            will be added to the plant. Otherwise, all model instances will be added.
            `add_weld` directives connecting added model instances to each other or to
            the world are also preserved.

        add_frozen_child_instances: If True and model_instance_names is not None, then
            model_instances that are not listed in model_instance_names, but are welded
            to a model_instance that is listed, will be added to the plant; with all
            joints replaced by welded joints.

        package_xmls: A list of package.xml file paths that will be passed to
            the parser, using Parser.AddPackageXml().

        parser_preload_callback: A callback function that will be called after
            the Parser is created, but before any directives are processed. This can be
            used to add additional packages to the parser, or to add additional model
            directives.

        parser_prefinalize_callback: A callback function that will be called
            after the directives are processed, but before the plant is finalized. This
            can be used to add additional model directives.

    Returns:
        A MultibodyPlant populated from (a subset of) the scenario.
    """
    print('MakeMultibodyPlant was called')
    plant = MultibodyPlant(time_step=scenario.plant_config.time_step)
    parser = Parser(plant)
    _PopulatePlantOrDiagram(
        plant,
        parser,
        scenario,
        model_instance_names,
        add_frozen_child_instances,
        package_xmls,
        parser_preload_callback,
        parser_prefinalize_callback,
    )
    return plant

def ConfigureParser(parser: Parser):
    """Add the `manipulation` module packages to the given Parser."""
    package_xml = get_resource_path('package.xml', arg_provides_ext=True)
    parser.package_map().AddPackageXml(filename=package_xml)


def _ApplyPrefinalizeDriverConfigSim(
    driver_config,  # See Scenario.model_drivers for typing
    model_instance_name: str,
    sim_plant: MultibodyPlant,
    directives: typing.List[ModelDirective],
    models_from_directives_map: typing.Mapping[str, typing.List[ModelInstanceInfo]],
    package_xmls: typing.List[str],
    builder: DiagramBuilder,
) -> None:
    if isinstance(driver_config, JointStiffnessDriver):
        model_instance = sim_plant.GetModelInstanceByName(model_instance_name)

        # Set PD gains.
        for name, gains in driver_config.gains.items():
            actuator = sim_plant.GetJointActuatorByName(name, model_instance)
            actuator.set_controller_gains(PdControllerGains(p=gains.kp, d=gains.kd))

        # Turn off gravity to model (perfect) gravity compensation.
        sim_plant.set_gravity_enabled(model_instance, False)
        if driver_config.hand_model_name:
            sim_plant.set_gravity_enabled(
                sim_plant.GetModelInstanceByName(driver_config.hand_model_name),
                False,
            )


def _ApplyPrefinalizeDriverConfigsSim(
    *,
    driver_configs,  # See Scenario.model_drivers for typing
    sim_plant: MultibodyPlant,
    directives: typing.List[ModelDirective],
    models_from_directives: typing.Mapping[str, typing.List[ModelInstanceInfo]],
    package_xmls: typing.List[str],
    builder: DiagramBuilder,
) -> None:
    models_from_directives_map = dict(
        [(info.model_name, info) for info in models_from_directives]
    )
    for model_instance_name, driver_config in driver_configs.items():
        _ApplyPrefinalizeDriverConfigSim(
            driver_config,
            model_instance_name,
            sim_plant,
            directives,
            models_from_directives_map,
            package_xmls,
            builder,
        )

def _ApplyDriverConfigSim(
    driver_config,  # See Scenario.model_drivers for typing
    model_instance_name: str,
    sim_plant: MultibodyPlant,
    scenario: Scenario,
    package_xmls: typing.List[str],
    builder: DiagramBuilder,
) -> None:
    if isinstance(driver_config, IiwaDriver):
        print('we adding IiwaDriver')
        model_instance = sim_plant.GetModelInstanceByName(model_instance_name)
        num_iiwa_positions = sim_plant.num_positions(model_instance)

        # Make the plant for the iiwa controller to use.
        controller_plant = MakeMultibodyPlant(
            scenario=scenario,
            model_instance_names=[model_instance_name],
            add_frozen_child_instances=True,
            package_xmls=package_xmls,
        )
        # Keep the controller plant alive during the Diagram lifespan.
        builder.AddNamedSystem(
            f"{model_instance_name}_controller_plant_pointer_system",
            SharedPointerSystem(controller_plant),
        )

        control_mode = ParseIiwaControlMode(driver_config.control_mode)
        sim_iiwa_driver = SimIiwaDriver.AddToBuilder(
            plant=sim_plant,
            iiwa_instance=model_instance,
            controller_plant=controller_plant,
            builder=builder,
            ext_joint_filter_tau=0.01,
            desired_iiwa_kp_gains=np.full(num_iiwa_positions, 100),
            control_mode=control_mode,
        )
        for i in range(sim_iiwa_driver.num_input_ports()):
            port = sim_iiwa_driver.get_input_port(i)
            if not builder.IsConnectedOrExported(port):
                builder.ExportInput(port, f"{model_instance_name}.{port.get_name()}")
        for i in range(sim_iiwa_driver.num_output_ports()):
            port = sim_iiwa_driver.get_output_port(i)
            builder.ExportOutput(port, f"{model_instance_name}.{port.get_name()}")

    elif isinstance(driver_config, SchunkWsgDriver):
        model_instance = sim_plant.GetModelInstanceByName(model_instance_name)
        # Wsg controller.
        wsg_controller = builder.AddSystem(SchunkWsgPositionController())
        wsg_controller.set_name(model_instance_name + ".controller")
        builder.Connect(
            wsg_controller.get_generalized_force_output_port(),
            sim_plant.get_actuation_input_port(model_instance),
        )
        builder.Connect(
            sim_plant.get_state_output_port(model_instance),
            wsg_controller.get_state_input_port(),
        )
        builder.ExportInput(
            wsg_controller.get_desired_position_input_port(),
            model_instance_name + ".position",
        )
        builder.ExportInput(
            wsg_controller.get_force_limit_input_port(),
            model_instance_name + ".force_limit",
        )
        wsg_mbp_state_to_wsg_state = builder.AddSystem(
            MakeMultibodyStateToWsgStateSystem()
        )
        builder.Connect(
            sim_plant.get_state_output_port(model_instance),
            wsg_mbp_state_to_wsg_state.get_input_port(),
        )
        builder.ExportOutput(
            wsg_mbp_state_to_wsg_state.get_output_port(),
            model_instance_name + ".state_measured",
        )
        builder.ExportOutput(
            wsg_controller.get_grip_force_output_port(),
            model_instance_name + ".force_measured",
        )

    elif isinstance(driver_config, InverseDynamicsDriver):
        model_instance_names = model_instance_name.split("+")
        model_instances = [
            sim_plant.GetModelInstanceByName(n) for n in model_instance_names
        ]

        controller_plant = MultibodyPlant(time_step=sim_plant.time_step())
        parser = Parser(controller_plant)
        for p in package_xmls:
            parser.package_map().AddPackageXml(p)
        ConfigureParser(parser)

        # Make the plant for the iiwa controller to use.
        controller_directives = []
        for d in FlattenModelDirectives(
            ModelDirectives(directives=scenario.directives), parser.package_map()
        ).directives:
            if d.add_model and (d.add_model.name in model_instance_names):
                controller_directives.append(d)
            if (
                d.add_weld
                and (
                    ScopedName.Parse(d.add_weld.child).get_namespace()
                    in model_instance_names
                )
                and (
                    d.add_weld.parent == "world"
                    or ScopedName.Parse(d.add_weld.parent).get_namespace()
                    in model_instance_names
                )
            ):
                controller_directives.append(d)
        ProcessModelDirectives(
            directives=ModelDirectives(directives=controller_directives),
            parser=parser,
        )
        controller_plant.Finalize()

        # Add the controller

        # When using multiple model instances, the model instance name must be prefixed.
        # The strings should take the form {model_instance_name}_{joint_actuator_name}, as
        # prescribed by MultiBodyPlant::GetActuatorNames().
        add_model_instance_prefix = len(model_instance_names) > 1
        actuator_names = controller_plant.GetActuatorNames(add_model_instance_prefix)

        # Check that all actuator names are valid.
        for actuator_name in driver_config.gains.keys():
            if actuator_name not in actuator_names:
                raise ValueError(
                    f"Actuator '{actuator_name}' not found. Valid names are: {actuator_names}"
                )

        # Get gains for each joint from the config. Use default gains if it doesn't exist in the config.
        default_gains = JointPidControllerGains()
        gains: typing.List[JointPidControllerGains] = []
        for actuator_name in actuator_names:
            joint_gains = driver_config.gains.get(actuator_name, default_gains)
            gains.append(joint_gains)

        controller = builder.AddSystem(
            InverseDynamicsController(
                controller_plant,
                kp=[joint_gains.kp for joint_gains in gains],
                ki=[joint_gains.ki for joint_gains in gains],
                kd=[joint_gains.kd for joint_gains in gains],
                has_reference_acceleration=False,
            )
        )
        controller.set_name(model_instance_name + ".controller")
        if len(model_instances) == 1:
            builder.Connect(
                sim_plant.get_state_output_port(model_instances[0]),
                controller.get_input_port_estimated_state(),
            )
            builder.Connect(
                controller.get_output_port(),
                sim_plant.get_actuation_input_port(model_instances[0]),
            )
            builder.ExportOutput(
                sim_plant.get_state_output_port(model_instances[0]),
                model_instance_name + ".state_estimated",
            )
        else:
            combined_state = builder.AddSystem(
                _MultiplexState(sim_plant, model_instance_names)
            )
            combined_state.set_name(model_instance_name + ".combined_state")
            combined_input = builder.AddSystem(
                _DemultiplexInput(sim_plant, model_instance_names)
            )
            combined_input.set_name(model_instance_name + ".combined_input")
            for index, model_instance in enumerate(model_instances):
                builder.Connect(
                    sim_plant.get_state_output_port(model_instance),
                    combined_state.get_input_port(index),
                )
                builder.Connect(
                    combined_input.get_output_port(index),
                    sim_plant.get_actuation_input_port(model_instance),
                )
            builder.Connect(
                combined_state.get_output_port(),
                controller.get_input_port_estimated_state(),
            )
            builder.Connect(
                controller.get_output_port(), combined_input.get_input_port()
            )
            builder.ExportOutput(
                combined_state.get_output_port(),
                model_instance_name + ".state_estimated",
            )

        builder.ExportInput(
            controller.get_input_port_desired_state(),
            model_instance_name + ".desired_state",
        )

    elif isinstance(driver_config, JointStiffnessDriver):
        model_instance = sim_plant.GetModelInstanceByName(model_instance_name)

        # PD gains and gravity comp are set in ApplyPrefinalizeDriverConfigsSim

        builder.ExportInput(
            sim_plant.get_desired_state_input_port(model_instance),
            model_instance_name + ".desired_state",
        )
        builder.ExportInput(
            sim_plant.get_actuation_input_port(model_instance),
            model_instance_name + ".tau_feedforward",
        )
        builder.ExportOutput(
            sim_plant.get_state_output_port(model_instance),
            model_instance_name + ".state_estimated",
        )

def _ApplyDriverConfigsSim(
    *,
    driver_configs,  # See Scenario.model_drivers for typing
    sim_plant: MultibodyPlant,
    scenario: Scenario,
    package_xmls: typing.List[str],
    builder: DiagramBuilder,
) -> None:
    for model_instance_name, driver_config in driver_configs.items():
        _ApplyDriverConfigSim(
            driver_config=driver_config,
            model_instance_name=model_instance_name,
            sim_plant=sim_plant,
            scenario=scenario,
            package_xmls=package_xmls,
            builder=builder,
        )

class TwoDArgs(Tap):
    target_browser_for_replay: str = 'xdg-open'
    diagram_destination: str = 'diagram.png'
    log_destination: str = '2d-log.json'
    select_controller: typing.Literal['stiffness', 'hybrid'] = 'stiffness'
    use_traj_vis: bool = False


def MakeHardwareStation(scenario: Scenario, meshcat: Meshcat, parser_prefinalize_callback: typing.Callable[[Parser], None] = None) -> Diagram:
    robot_builder = RobotDiagramBuilder(time_step=TIME_STEP)
    builder = robot_builder.builder()
    sim_plant = robot_builder.plant()
    scene_graph = robot_builder.scene_graph()
    ApplyMultibodyPlantConfig(scenario.plant_config, sim_plant)

    parser = Parser(sim_plant)
    ConfigureParser(parser)

    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        parser=parser,
    )

    if parser_prefinalize_callback:
        parser_prefinalize_callback(parser)

    _ApplyPrefinalizeDriverConfigsSim(
        driver_configs=scenario.model_drivers,
        sim_plant=sim_plant,
        directives=scenario.directives,
        models_from_directives=added_models,
        package_xmls=[],
        builder=builder,
    )

    sim_plant.Finalize()

    cparams = ContactVisualizerParams()
    cparams.force_threshold = 1e-6
    cparams.moment_threshold = 1e-4
    cparams.newtons_per_meter = 200.0
    cparams.radius = 0.004
    cparams.color=Rgba(.1, .9, .1, .9)

    contact_visualizer = ContactVisualizer.AddToBuilder(
        builder, sim_plant, meshcat, cparams
    )

    # For some Apply* functions in this workflow, we _never_ want LCM.
    scenario.lcm_buses["opt_out"] = DrakeLcmParams(lcm_url="memq://null")

    ## Add LCM buses. (The simulator will handle polling the network for new
    ## messages and dispatching them to the receivers, i.e., "pump" the bus.)
    lcm_buses = ApplyLcmBusConfig(lcm_buses=scenario.lcm_buses, builder=builder)

    _ApplyDriverConfigsSim(
        driver_configs=scenario.model_drivers,
        sim_plant=sim_plant,
        scenario=scenario,
        package_xmls=[],
        builder=builder,
    )

    # Add visualization.
    if meshcat:
        ApplyVisualizationConfig(
            scenario.visualization, builder, meshcat=meshcat, lcm_buses=lcm_buses
        )

    # Export "cheat" ports.
    builder.ExportInput(
        sim_plant.get_applied_generalized_force_input_port(),
        "applied_generalized_force",
    )
    builder.ExportInput(
        sim_plant.get_applied_spatial_force_input_port(),
        "applied_spatial_force",
    )

    # Export any actuation (non-empty) input ports that are not already
    # connected (e.g. by a driver).
    for i in range(sim_plant.num_model_instances()):
        port = sim_plant.get_actuation_input_port(ModelInstanceIndex(i))
        if port.size() > 0 and not builder.IsConnectedOrExported(port):
            builder.ExportInput(port, port.get_name())
    # Export all MultibodyPlant output ports.
    for i in range(sim_plant.num_output_ports()):
        builder.ExportOutput(
            sim_plant.get_output_port(i),
            sim_plant.get_output_port(i).get_name(),
        )
    # Export the only SceneGraph output port.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")

    station_diagram = robot_builder.Build()
    station_diagram.set_name('station')
    return station_diagram


def Setup(parser):
    parser.plant().set_discrete_contact_approximation(
        DiscreteContactApproximation.kTamsi
        # kLagged
        # kTamsi
        # kSap
    )


def simulate_2d(args: TwoDArgs):
    meshcat = StartMeshcat()
    # meshcat.Set2dRenderMode(xmin=-0.25, xmax=1.5, ymin=-0.1, ymax=1.3)

    builder = DiagramBuilder()

    scenario = LoadScenario(filename=get_resource_path('planar_manipulation_station.scenario.yaml', arg_provides_ext=True))
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat, parser_prefinalize_callback=Setup))

    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")

    visualizer = None
    if args.use_traj_vis:
        visualizer = station.GetSubsystemByName("meshcat_visualizer(illustration)")

    adder = builder.AddSystem(Adder(2, 3))
    controller = builder.AddSystem(TrajFollowingJointStiffnessController(plant, 100., 20.))
    builder.ExportInput(controller.GetInputPort('trajectory'), 'inner_trajectory')
    builder.ExportInput(controller.GetInputPort('switched_on_intervals'), 'stiff_c_switched_on_intervals')
    force_sensor = builder.AddSystem(ForceSensor(plant))

    wsg_trajectory_source = builder.AddSystem(TrajectorySource(make_dummy_wsg_trajectory()))
    builder.Connect(wsg_trajectory_source.get_output_port(), station.GetInputPort('wsg.position'))

    if 'stiffness' == args.select_controller:
        builder.ExportInput(adder.get_input_port(1), 'torque_adder_2nd_term')
    elif 'hybrid' == args.select_controller:
        hyb_controller = builder.AddSystem(HybridCartesianController(plant, 100., 20., 0.2, 10.))
        builder.ExportInput(hyb_controller.GetInputPort('switched_on_intervals'), 'hyb_c_switched_on_intervals')
        builder.ExportInput(hyb_controller.GetInputPort('trajectory'), 'cartesian_trajectory')

        builder.Connect(
            force_sensor.GetOutputPort("sensed_force_out"),
            hyb_controller.GetInputPort("ee_force_measured"),
        )

        builder.Connect(
            station.GetOutputPort('body_poses'),
            hyb_controller.GetInputPort('body_poses')
        )
        builder.Connect(
            station.GetOutputPort('body_spatial_velocities'),
            hyb_controller.GetInputPort('body_spatial_velocities')
        )

        builder.Connect(
            station.GetOutputPort("iiwa_state"),
            hyb_controller.GetInputPort("iiwa_state_measured"),
        )

        builder.Connect(
            hyb_controller.GetOutputPort('iiwa_torque_cmd'),
            adder.get_input_port(1)
        )

    else:
        raise Exception('unknown controller requested')

    builder.Connect(
        station.GetOutputPort('iiwa_generalized_contact_forces'),
        force_sensor.GetInputPort("iiwa_inner_forces_in"),
    )
    builder.Connect(
        station.GetOutputPort('iiwa_state'),
        force_sensor.GetInputPort("iiwa_state_measured"),
    )
    builder.Connect(
        station.GetOutputPort('body_poses'),
        force_sensor.GetInputPort('body_poses')
    )

    builder.Connect(
        station.GetOutputPort("iiwa_state"),
        controller.GetInputPort("iiwa_state_measured"),
    )
    builder.Connect(
        controller.GetOutputPort('iiwa_torque_cmd'),
        adder.get_input_port(0)
    )
    builder.Connect(
        adder.get_output_port(),
        station.GetInputPort("iiwa_actuation"),
    )

    diagram = builder.Build()
    pydot.graph_from_dot_data(diagram.GetGraphvizString(max_depth=4))[0].write_png(args.diagram_destination)

    if not args.use_traj_vis:
        simulator = Simulator(diagram)
        global_context = simulator.get_mutable_context()
    else:
        global_context = diagram.CreateDefaultContext()

    plant_context = plant.GetMyContextFromRoot(global_context)
    plant.SetPositions(
        plant_context,
        plant.GetModelInstanceByName("iiwa"),
        INITIAL_IIWA_COORDS,
    )

    X_WG = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName('body'))
    X_WVstart = plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName('nut'))

    AddMeshcatTriad(meshcat, 'NutInitPose', X_PT=X_WVstart)


    R_GoalGripper = RotationMatrix(RollPitchYaw(np.radians([90, 0, 90]))).inverse()
    t_GoalGripper = [0.02, -0.075, 0.]
    t_GoalGripperPreGrasp = [0.02, -0.225, 0.]
    X_GoalGripper = RigidTransform(R_GoalGripper, [.01]*3)
    X_gripper_offset = RigidTransform(RotationMatrix.Identity(), t_GoalGripper)
    X_pre_grasp_offset = RigidTransform(RotationMatrix.Identity(), t_GoalGripperPreGrasp)

    X_WV_G = X_WVstart @ X_GoalGripper
    AddMeshcatTriad(meshcat, 'NutInitPoseInG', X_PT=X_WV_G)

    possible_nut_rotations = []
    for ind, alpha in enumerate(np.radians(np.arange(0, 360, 60))):
        R_1 = RollPitchYaw([0, alpha, 0]).ToRotationMatrix() @ X_WVstart.rotation()
        vec = np.degrees(RollPitchYaw(R_1).vector()).tolist()
        vec_with_ind = [ind, np.round(np.degrees(alpha))] + vec
        #print("// ind={} angle {} RollPitchYaw: {:.1f} {:.1f} {:.1f}".format(*vec_with_ind))
        #AddMeshcatTriad(meshcat, f'nut-sides-{alpha:.3f}', X_PT=X_curr_rot, opacity=0.33)
        possible_nut_rotations.append(R_1)

    arg_min = 0
    least_ang_distance = float('inf')
    R_WNutInBetterOrientation = RotationMatrix(RollPitchYaw(np.radians([90., 30., 0.])))
    for ind, R_WNcand in enumerate(possible_nut_rotations):
        loss = np.linalg.norm(np.degrees(RollPitchYaw(R_WNcand.InvertAndCompose(R_WNutInBetterOrientation)).vector()))
        if loss < least_ang_distance:
            least_ang_distance = loss
            arg_min = ind

    X_WGripperAtTurnStart_ = RigidTransform(possible_nut_rotations[arg_min], X_WVstart.translation()) @ X_GoalGripper
    R_WVend_ = RollPitchYaw(np.radians([0, 30, 0])).ToRotationMatrix() @ possible_nut_rotations[arg_min]
    X_WGripperAtTurnEnd_ = RigidTransform(R_WVend_, X_WVstart.translation()) @ X_GoalGripper

    X_WGripperAtTurnStart = X_WGripperAtTurnStart_ @ X_gripper_offset
    X_WGripperAtTurnEnd = X_WGripperAtTurnEnd_ @ X_gripper_offset

    X_WGripperPreGraspAtTurnStart = X_WGripperAtTurnStart_ @ X_pre_grasp_offset
    X_WGripperPostGraspAtTurnEnd = X_WGripperAtTurnEnd_ @ X_pre_grasp_offset

    AddMeshcatTriad(meshcat, 'pregrasp-at-initial-valve', X_PT=X_WGripperPreGraspAtTurnStart)
    AddMeshcatTriad(meshcat, 'gripper-at-initial-valve', X_PT=X_WGripperAtTurnStart)
    AddMeshcatTriad(meshcat, 'gripper-at-final-valve', X_PT=X_WGripperAtTurnEnd)
    AddMeshcatTriad(meshcat, 'postgrasp-at-final-valve', X_PT=X_WGripperPostGraspAtTurnEnd)

    trajectory, ts = optimize_target_trajectory([X_WG, X_WGripperPreGraspAtTurnStart, X_WGripperAtTurnStart, X_WGripperAtTurnEnd, X_WGripperPostGraspAtTurnEnd, X_WG],
                                                plant, plant_context)
    if trajectory is None and not args.use_traj_vis:
        print('opt didnt succeed')
        return

    diagram.GetInputPort('inner_trajectory').FixValue(global_context, trajectory)

    if not args.use_traj_vis:
        wsg_trajectory = make_wsg_trajectory(ts)
        wsg_trajectory_source.UpdateTrajectory(wsg_trajectory)

    if 'stiffness' == args.select_controller and not args.use_traj_vis:
        diagram.GetInputPort('torque_adder_2nd_term').FixValue(global_context, np.zeros((3),))
        diagram.GetInputPort('stiff_c_switched_on_intervals').FixValue(global_context, np.array([[ts[0], ts[-1]]]))

    elif 'hybrid' == args.select_controller and not args.use_traj_vis:
        cart_trajectory = make_cartesian_trajectory([X_WGripperAtTurnStart, X_WGripperAtTurnEnd], [ts[3], ts[4]])
        diagram.GetInputPort('cartesian_trajectory').FixValue(global_context, cart_trajectory)

        diagram.GetInputPort('stiff_c_switched_on_intervals').FixValue(global_context, np.array([[ts[0], ts[3]],
                                                                                                 [ts[4], ts[-1]]]))
        diagram.GetInputPort('hyb_c_switched_on_intervals').FixValue(global_context, np.array([[ts[3], ts[4]]]))

    # minimalist_traj_vis(trajectory)
    #meshcat.SetObject("start", Sphere(0.03), rgba=Rgba(.9, .1, .1, .7))
    #meshcat.SetTransform("start", X_Wpstart)
    #meshcat.SetObject("end", Sphere(0.03), rgba=Rgba(.1, .9, .1, .7))
    #meshcat.SetTransform("end", X_Wpend)

    if args.use_traj_vis:
        rich_trajectory_vis(trajectory, global_context, plant, visualizer, meshcat)
    else:
        state_monitor = StateMonitor(args.log_destination, plant, diagram,
                                     trajectory if 'stiffness' == args.select_controller else None,
                                     cart_trajectory if 'hybrid' == args.select_controller else None,
                                     meshcat)
        simulator.set_monitor(state_monitor.callback)

        plant.SetPositions(plant_context, plant.GetModelInstanceByName("iiwa"), trajectory.value(0))

        simulator.Initialize()
        meshcat.StartRecording(set_visualizations_while_recording=False)
        simulator.AdvanceTo(trajectory.end_time())
        meshcat.PublishRecording()

    web_url = meshcat.web_url()
    print(f'Meshcat is now available at {web_url}')
    os.system(f'{args.target_browser_for_replay} {web_url}')
    time.sleep(30)


if '__main__' == __name__:
    simulate_2d(TwoDArgs().parse_args())
