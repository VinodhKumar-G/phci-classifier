from pathlib import Path
import sys

# Ensure root-level imports work from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))

from dashboard.mqtt_client import MQTTPublisher

SPECIES = ['mimosa', 'tomato', 'aloe']
STRESSES = ['healthy', 'drought', 'heat']
SP_IDX = {name: idx for idx, name in enumerate(SPECIES)}
ST_IDX = {name: idx for idx, name in enumerate(STRESSES)}

__all__ = ['MQTTPublisher', 'SP_IDX', 'ST_IDX', 'SPECIES', 'STRESSES']
