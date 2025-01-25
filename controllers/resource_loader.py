import dataclasses as dc
import os
import sys
import typing

import numpy as np

from pydrake.all import (
    CameraConfig,
    Parser,
    RevoluteJoint,
    RigidTransform,
    RollPitchYaw,
    MultibodyPlant,
    SimulatorConfig,
    MultibodyPlantConfig,
    ModelDirective,
    DrakeLcmParams,
    IiwaDriver,
    SchunkWsgDriver,
    ZeroForceDriver,
    VisualizationConfig,
)

from pydrake.common.yaml import yaml_load_typed

TIME_STEP=1.e-4  # faster

def get_resource_path(resource_name: str, arg_provides_ext=True) -> str:
    ext = '' if arg_provides_ext else '.sdf'
    resource_file = os.path.join('resources', f'{resource_name}{ext}')
    full_path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), resource_file)
    if not os.path.exists(full_path) or not os.path.isfile(full_path):
        raise Exception(f'a resource {resource_name} is absent at {full_path}')
    return full_path


def get_resource(plant: MultibodyPlant, resource_name: str):
    resource_path = get_resource_path(resource_name)
    resource_model = Parser(plant=plant).AddModels(resource_path)[0]
    return resource_model

@dc.dataclass
class Scenario:
    """Defines the YAML format for a (possibly stochastic) scenario to be
    simulated.

    Args:
        random_seed: Random seed for any random elements in the scenario. The
            seed is always deterministic in the `Scenario`; a caller who wants
            randomness must populate this value from their own randomness.
        simulation_duration: The maximum simulation time (in seconds).  The
            simulator will attempt to run until this time and then terminate.
        simulator_config: Simulator configuration (integrator and publisher
            parameters).
        plant_config: Plant configuration (time step and contact parameters).
        directives: All of the fully deterministic elements of the simulation.
        lcm_buses: A map of {bus_name: lcm_params} for LCM transceivers to be
            used by drivers, sensors, etc.
        model_drivers: For actuated models, specifies where each model's
            actuation inputs come from, keyed on the ModelInstance name.
        cameras: Cameras to add to the scene (and broadcast over LCM). The key
            for each camera is a helpful mnemonic, but does not serve a
            technical role. The CameraConfig::name field is still the name that
            will appear in the Diagram artifacts.
        visualization: Visualization configuration.
    """

    random_seed: int = 0

    simulation_duration: float = np.inf

    simulator_config: SimulatorConfig = SimulatorConfig(
        max_step_size=0.01,
        use_error_control=False,
        accuracy=1.0e-2,
    )

    plant_config: MultibodyPlantConfig = MultibodyPlantConfig(
        discrete_contact_approximation="tamsi",
        time_step=TIME_STEP,
    )

    directives: typing.List[ModelDirective] = dc.field(default_factory=list)

    # Opt-out of LCM by default.
    lcm_buses: typing.Mapping[str, DrakeLcmParams] = dc.field(
        default_factory=lambda: dict(default=DrakeLcmParams(lcm_url="memq://null"))
    )

    model_drivers: typing.Mapping[
        str,
        typing.Union[
            IiwaDriver,
            #InverseDynamicsDriver,
            #JointStiffnessDriver,
            SchunkWsgDriver,
            ZeroForceDriver,
        ],
    ] = dc.field(default_factory=dict)

    cameras: typing.Mapping[str, CameraConfig] = dc.field(default_factory=dict)

    camera_ids: typing.Mapping[str, str] = dc.field(default_factory=dict)

    visualization: VisualizationConfig = VisualizationConfig(publish_contacts=False)

# TODO(russt): load from url (using packagemap).
def LoadScenario(
    *,
    filename: str = None,
    data: str = None,
    scenario_name: str = None,
    defaults: Scenario = Scenario(),
) -> Scenario:
    """Implements the command-line handling logic for scenario data.

    Args:
        filename: A yaml filename to load the scenario from.

        data: A yaml _string_ to load the scenario from. If both filename and string are
            specified, then the filename is parsed first, and then the string is _also_
            parsed, potentially overwriting defaults from the filename. Note: this will
            not append additional `directives`, it will replace them; see
            AppendDirectives.

        scenario_name: The name of the scenario/child to load from the yaml file. If
            None, then the entire file is loaded.

        defaults: A `Scenario` object to use as the default values.

    Returns:
        A `Scenario` object loaded from the given input arguments.
    """
    result = defaults
    if filename:
        result = yaml_load_typed(
            schema=Scenario,
            filename=filename,
            child_name=scenario_name,
            defaults=result,
            retain_map_defaults=True,
        )
    if data:
        result = yaml_load_typed(
            schema=Scenario,
            data=data,
            child_name=scenario_name,
            defaults=result,
            retain_map_defaults=True,
        )
    return result
