"""
In this config, the previous sensor mapping issue (OW and OQ being swapped) has been fixed. Additionally, the OP sensor was found to be unplugged and has now been resolved. This likely dates back to the sensor change from triaxial 4I -> OP (2025-02-20).

"""
from dataclasses import dataclass
import numpy as np

axis_mask = np.array([
    [0, 1, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 0],
    [0, 1, 1],
    [1, 1, 0],
    [0, 1, 1]
])

from fmcg.utils.data import generate_array_coordinates
r_sensors = generate_array_coordinates(grid_shape=(4, 4), grid_spacing=0.0206, y=0)


@dataclass
class SystemConfig:
    """
    SystemConfiguration class holds the configuration details for the sensor system.

    Attributes:
        gen3quspin_coor_to_plate_coor (list of list of int):
            Represents the orientation of the sensors in the holder by mapping the coordinate system of a Gen3 sensor to
            the coordinate system of the holder. Each entry represents the sensors in descending order. Each sensor is
            represented by a list of 3 integers, where the value represents the axis in the Gen3 sensor coordinate
            system [1->x, 2->y, 3->z] and the index/position represents the axis in the holder coordinate system [x, y,
            z]. The sign of the value represents the direction of the axis. Example [-1,-3,2]: The y-axis of the holder
            is given by the negative z-axis of a Gen3 QuSpin sensor at that position in the grid.

        is_gen2 (dict):
            A dictionary indicating whether a sensor is a Gen2 sensor. The keys are sensor identifiers and the values
            are booleans indicating if the sensor is a Gen2 sensor.

        gen3_to_gen2 (list of int):
            Coordinate system mapping for Gen2 sensors. Each value represents the axis in the Gen3 sensor coordinate
            system [1->x, 2->y, 3->z] and the index/position represents the axis in the Gen2 sensor coordinate system
            [x, y, z]. The sign of the value represents the direction of the axis.
    """

    # Represents the orientation of the sensors in the holder by mapping the coordinate system of a Gen3 sensor to the
    # coordinate system of the holder. Each entry represents the sensors in descending order. Each sensor is represented
    # by a list of 3 integers, where the value represents the axis in the Gen3 sensor coordinate system [1->x, 2->y,
    # 3->z] and the index / position represents the axis in the holder coordinate system [x, y, z]. The sign of the
    # value represents the direction of the axis. Example [-1,-3,2]: The y-axis of the holdeer is given by neg. z-axis
    # of a gen3 quspin at that position in the grid.
    gen3quspin_coor_to_plate_coor = [
        [1, -3, -2],    #OQ
        [2, -3, -1],     #F1
        [1, -3, -2],    #OO
        [-2, -3, -1],     #OX
        [2, -3, 1],   #OW
        [1, -3, -2],    #OY
        [-2, -3, -1],   #EZ
        [1, -3, -2],    #YQ, GEN2
        [-1, -3, 2],    #OT
        [2, -3, 1],     #OP
        [-1, -3, 2],    #F0
        [-2, -3, -1],     #C1, GEN2
        [2, -3, 1],   #EY
        [-1, -3, 2],    #YP, GEN2
        [-2, -3, -1],   #OR
        [-1, -3, 2],    #F2
    ]

    # Whether a sensor is a Gen2 sensor
    is_gen2 = {
        "OW": False, 
        "OT": False, # has often problems
        "F1": False, # has often problems
        "EY": False,
        "EZ": False, 
        "OX": False,
        "OR": False, 
        "OQ": False, 
        "C1": False, #True # has often problems
        "YP": False, #True # has often problems
        "YQ": False, #True # has often problems
        "OO": False, 
        "F0": False, # has often problems
        "F2": False,
        "OY": False,
        "NL": False,
        "OP": False,
    }
    # Coordinate system mapping for Gen2 sensors
    gen3_to_gen2 = [-1, -1, 1]


@dataclass
class MeasurementConfig:
    """
    MeasurementConfiguration class holds the configuration for the measurement setup.

    Attributes:
        quspin_positions (list of str): The position of the sensors in the holder (4x4 grid).
            Example: ["F1", "OX", "OU", "OQ", "YQ", "EZ", "OY", "OW", "OT", "NL", "F0", "C1", "EY", "YP", "OR", "F2"]

        DAC1 (str): The name of the first DAC.
            Example: "cDAQ1Mod1/ai"

        DAC2 (str): The name of the second DAC.
            Example: "Dev1/ai"

        quspin_mapping (dict): The mapping of the DAC channels to the sensors and their axes.
            The format is {sensor_name: [X_channel, Y_channel, Z_channel]}. Example: {
                "YP": [None, "cDAQ1Mod1/ai0", "cDAQ1Mod1/ai1"], "C1": [None, "cDAQ1Mod1/ai2", "cDAQ1Mod1/ai3"], ...
                "NL": ["Dev1/ai1", "Dev1/ai2", "Dev1/ai3"]

        axis_mask (ndarray): (n_sensors, 3) int array indicating which field components are
            physically observed. 1 = observed, 0 = unobserved (channel not connected).
            Row order matches quspin_positions (row-major, top-left to bottom-right).
            Derived from quspin_mapping: None entries → 0, connected entries → 1.
    """

    # Which field components are observed per sensor; 1 = observed, 0 = unobserved.
    # Row order matches quspin_positions (row-major). Mirrors None entries in quspin_mapping.
    axis_mask = axis_mask

    # The position of the sensors in the holder (4x4 grid) ordering 1-16
    quspin_positions = [
        ["OQ", "F1", "OO", "OX"],
        ["OW", "OY", "EZ", "YQ"],
        ["OT", "OP", "F0", "C1"],
        ["EY", "YP", "OR", "F2"],
    ]
    # The names of the DAC
    DAC1 = "cDAQ1Mod1/ai"
    DAC2 = "Dev1/ai"
    # The mapping of the DAC channels to the sensors and their axes in the format: {sensor_name: [X_channel, Y_cannel,
    # Z_channel]}
    quspin_mapping = {
        "YP": [None, f"{DAC1}0", f"{DAC1}1"],
        "C1": [None, f"{DAC1}2", f"{DAC1}3"],
        "YQ": [None, f"{DAC1}4", f"{DAC1}5"],
        "OQ": [None, f"{DAC1}6", f"{DAC1}7"],
        "OY": [None, f"{DAC1}16", f"{DAC1}17"],
        "F2": [None, f"{DAC1}18", f"{DAC1}19"],
        "F0": [None, f"{DAC1}20", f"{DAC1}21"],
        "OX": [None, f"{DAC1}22", f"{DAC1}23"],
        "OW": [None, f"{DAC1}10", f"{DAC1}11"],
        "OT": [None, f"{DAC1}12", f"{DAC1}13"],
        "F1": [None, f"{DAC1}14", f"{DAC1}15"],
        "EY": [None, f"{DAC1}24", f"{DAC1}25"],
        "EZ": [None, f"{DAC1}26", f"{DAC1}27"],
        "OO": [None, f"{DAC1}28", f"{DAC1}29"],
        "OR": [None, f"{DAC1}30", f"{DAC1}31"],
        "OP": [f"{DAC2}1", f"{DAC2}2", f"{DAC2}3"],
    }
