import numpy as np

"""Singleton class used to store meaningful information.
"""


class Parameters:
    _instance = None

    # Original sensors coordinates
    original_sensors = []
    original_x_coords = []
    original_y_coords = []
    original_z_coords = []

    # DB sensors coordinates
    filtered_sensors = []
    x_coords = []
    y_coords = []
    z_coords = []

    # Columns names
    weather_columns = []
    watering_columns = []
    original_sensor_columns = []
    sensor_columns = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Parameters, cls).__new__(cls)
        return cls._instance

    def set_params(cls, original_sensors_coords, sensors_coords):
        cls.original_sensors = original_sensors_coords
        cls.original_x_coords = list(set(np.array(cls.original_sensors)[:, 0]))
        cls.original_x_coords.sort()
        cls.original_y_coords = list(set(np.array(cls.original_sensors)[:, 1]))
        cls.original_y_coords.sort()
        cls.original_z_coords = list(set(np.array(cls.original_sensors)[:, 2]))
        cls.original_z_coords.sort(reverse=True)

        cls.filtered_sensors = sensors_coords
        cls.x_coords = list(set(np.array(cls.filtered_sensors)[:, 0]))
        cls.x_coords.sort()
        cls.y_coords = list(set(np.array(cls.filtered_sensors)[:, 1]))
        cls.y_coords.sort()
        cls.z_coords = list(set(np.array(cls.filtered_sensors)[:, 2]))
        cls.z_coords.sort(reverse=True)

        cls.weather_columns = [
            "air_temperature",
            "air_humidity",
            "wind_speed",
            "solar_radiation",
        ]
        cls.watering_columns = ["precipitation", "irrigation"]
        cls.original_sensor_columns = [
            f"z{z}_y{y}_x{x}"
            for z in cls.original_z_coords
            for y in cls.original_y_coords
            for x in cls.original_x_coords
        ]
        cls.sensor_columns = [
            f"z{z}_y{y}_x{x}"
            for z in cls.z_coords
            for y in cls.y_coords
            for x in cls.x_coords
        ]

    def get_original_sensors(cls):
        return cls.original_sensors

    def get_original_x_coords(cls):
        return cls.original_x_coords

    def get_original_y_coords(cls):
        return cls.original_y_coords

    def get_original_z_coords(cls):
        return cls.original_z_coords

    def get_filtered_sensors(cls):
        return cls.filtered_sensors

    def get_x_coords(cls):
        return cls.x_coords

    def get_y_coords(cls):
        return cls.y_coords

    def get_z_coords(cls):
        return cls.z_coords

    def get_weather_columns(cls):
        return cls.weather_columns

    def get_watering_columns(cls):
        return cls.watering_columns

    def get_original_sensor_columns(cls):
        return cls.original_sensor_columns

    def get_sensor_columns(cls):
        return cls.sensor_columns
