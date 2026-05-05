# src/dashboard/mqtt_client.py
# CORRECTED for paho-mqtt 2.x API

import os, json, time, logging
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion  # New in paho 2.x
from dotenv import load_dotenv

load_dotenv()  # Load from .env in local dev; GitHub Secrets in Codespaces

BROKER   = os.getenv('MQTT_BROKER_URL', 'mqtt.thingspeak.com')
PORT     = int(os.getenv('MQTT_PORT', '1883'))
USERNAME = os.getenv('MQTT_USERNAME', '')
PASSWORD = os.getenv('MQTT_PASSWORD', '')
CHANNEL  = os.getenv('THINGSPEAK_CHANNEL_ID', '0')

log = logging.getLogger(__name__)

# ─── paho-mqtt 2.x callbacks — MUST have 5 arguments ─────────────────

def _on_connect(client, userdata, flags, reason_code, properties):
    '''5-arg signature required by paho-mqtt 2.x (CallbackAPIVersion.VERSION2).'''
    if reason_code.is_failure:
        log.error(f'MQTT connect failed: {reason_code}')
    else:
        log.info(f'MQTT connected to {BROKER}')

def _on_publish(client, userdata, mid, reason_code, properties):
    '''5-arg signature required by paho-mqtt 2.x.'''
    if reason_code and reason_code.is_failure:
        log.warning(f'MQTT publish failed: mid={mid} reason={reason_code}')

def _on_disconnect(client, userdata, flags, reason_code, properties):
    '''5-arg signature required by paho-mqtt 2.x.'''
    if reason_code.value != 0:
        log.warning(f'MQTT unexpected disconnect: {reason_code}')

class MQTTPublisher:
    '''
    Publishes classification results to ThingSpeak via MQTT.
    Compatible with paho-mqtt 2.x (CallbackAPIVersion.VERSION2).
    '''

    def __init__(self):
        # REQUIRED in paho 2.x: pass callback_api_version explicitly
        self.client = mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=f'phci_{int(time.time())}',
            clean_session=True,
        )
        self.client.username_pw_set(USERNAME, PASSWORD)
        self.client.on_connect    = _on_connect
        self.client.on_publish    = _on_publish
        self.client.on_disconnect = _on_disconnect
        self._connected = False

    def connect(self) -> bool:
        try:
            self.client.connect(BROKER, PORT, keepalive=60)
            self.client.loop_start()
            time.sleep(0.5)  # Allow connection to establish
            self._connected = True
            return True
        except Exception as e:
            log.error(f'MQTT connect error: {e}')
            return False

    def publish_result(self, species: str, stress: str,
                       s1_conf: float, s2_conf: float,
                       voltage_mv: float) -> bool:
        if not self._connected:
            return False
        # ThingSpeak topic format: channels/{id}/publish
        topic = f'channels/{CHANNEL}/publish'
        payload = (
            f'field1={["mimosa","tomato","aloe"].index(species)}'
            f'&field2={["healthy","drought","heat"].index(stress)}'
            f'&field3={round(s1_conf*100,1)}'
            f'&field4={round(s2_conf*100,1)}'
            f'&field5={round(voltage_mv,3)}'
        )
        result = self.client.publish(topic, payload, qos=0)
        return result.rc == 0

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
        self._connected = False
