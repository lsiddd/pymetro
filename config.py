# ./config.py
from typing import Dict, List

# Game Configuration
class CONFIG:
    STATION_RADIUS: int = 15
    PASSENGER_SIZE: int = 6
    TRAIN_WIDTH: int = 25
    TRAIN_HEIGHT: int = 15
    LINE_WIDTH: int = 6
    MAX_PASSENGERS_PER_STATION: int = 6
    MAX_PASSENGERS_WITH_INTERCHANGE: int = 12
    TRAIN_CAPACITY: int = 6
    MAX_TRAINS_PER_LINE: int = 4
    MAX_CARRIAGES_PER_TRAIN: int = 3
    TRAIN_MAX_SPEED: float = 1.2
    TRAIN_ACCELERATION: float = 0.002
    PASSENGER_BOARD_TIME: int = 200
    INTERCHANGE_TRANSFER_TIME: int = 50  # Fast transfers at interchanges
    REGULAR_TRANSFER_TIME: int = 400     # Slow transfers at regular stations
    PASSENGER_WAIT_PATIENCE: int = 10000  # Time before reconsidering route
    BASE_SPAWN_RATE: int = 2000
    BASE_STATION_SPAWN_RATE: int = 15000
    DIFFICULTY_SCALE_FACTOR: float = 0.8
    OVERCROWD_TIME: int = 30000
    WEEK_DURATION: int = 60000
    LINE_COLORS: List[str] = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71', '#9b59b6', '#1abc9c', '#e67e22']
    CITIES: Dict[str, Dict[str, int]] = {
        'london': {'bridges': 2, 'maxLines': 5},
        'paris': {'bridges': 3, 'maxLines': 6},
        'newyork': {'bridges': 2, 'maxLines': 7},
        'tokyo': {'bridges': 4, 'maxLines': 6}
    }

# Station Types
class STATION_TYPES:
    CIRCLE: str = 'circle'
    TRIANGLE: str = 'triangle'
    SQUARE: str = 'square'
    PENTAGON: str = 'pentagon'
    DIAMOND: str = 'diamond'
    STAR: str = 'star'
    CROSS: str = 'cross'
    
    @classmethod
    def all_types(cls) -> List[str]:
        return [cls.CIRCLE, cls.TRIANGLE, cls.SQUARE, cls.PENTAGON, cls.DIAMOND, cls.STAR, cls.CROSS]
    
    @classmethod  
    def basic_types(cls) -> List[str]:
        return [cls.CIRCLE, cls.TRIANGLE, cls.SQUARE]
    
    @classmethod
    def special_types(cls) -> List[str]:
        return [cls.PENTAGON, cls.DIAMOND, cls.STAR, cls.CROSS]