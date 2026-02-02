import time
import sys
import json
import network
from umqtt.simple import MQTTClient
import config


class MQTTHandler:
    
    def __init__(self):
        self.client = None
        self.connected = False
        self.last_publish_time = 0
        self.messages_sent = 0
        self.connection_attempts = 0
        
        if config.MQTT_ENABLED:
            self._check_wifi_and_connect()
    
    def _check_wifi_and_connect(self):
        wlan = network.WLAN(network.STA_IF)
        
        if not wlan.active():
            sys.stdout.write("[MQTT] WiFi not active\n")
            return
        
        if not wlan.isconnected():
            sys.stdout.write("[MQTT] WiFi not connected\n")
            return
        
        sys.stdout.write(f"[MQTT] WiFi OK - IP: {wlan.ifconfig()[0]}\n")
        self._connect()
    
    def _connect(self):
        try:
            self.connection_attempts += 1
            
            sys.stdout.write("[MQTT] Connecting...\n")
            sys.stdout.write(f"[MQTT] Broker: {config.MQTT_BROKER}:{config.MQTT_PORT}\n")
            sys.stdout.write(f"[MQTT] Client ID: {config.MQTT_CLIENT_ID}\n")
            sys.stdout.write(f"[MQTT] Topic: {config.MQTT_TOPIC}\n")
            
            self.client = MQTTClient(
                config.MQTT_CLIENT_ID,
                config.MQTT_BROKER,
                port=config.MQTT_PORT,
                keepalive=config.MQTT_KEEPALIVE
            )
            
            self.client.connect()
            self.connected = True
            
            sys.stdout.write("[MQTT] Connected!\n")
            
        except OSError as e:
            sys.stdout.write(f"[MQTT] Network error: {e}\n")
            self.connected = False
        except Exception as e:
            sys.stdout.write(f"[MQTT] Error: {e}\n")
            self.connected = False
    
    def publish_status(self, light_state, countdown, vehicle_count, density):
        if not config.MQTT_ENABLED:
            return False
        
        if not self.connected:
            return False
        
        now = time.time()
        if now - self.last_publish_time < config.MQTT_UPDATE_INTERVAL:
            return False
        
        try:
            data = {
                "light": light_state,
                "countdown": countdown,
                "vehicles": vehicle_count,
                "density": density,
                "timestamp": int(now)
            }
            
            payload = json.dumps(data)
            self.client.publish(config.MQTT_TOPIC, payload)
            
            self.messages_sent += 1
            self.last_publish_time = now
            
            if config.DEBUG and self.messages_sent % 20 == 0:
                sys.stdout.write(f"[MQTT] Sent {self.messages_sent} messages\n")
            
            return True
            
        except OSError as e:
            sys.stdout.write(f"[MQTT] Publish failed: {e}\n")
            self.connected = False
            return False
        except Exception as e:
            sys.stdout.write(f"[MQTT] Error: {e}\n")
            self.connected = False
            return False
    
    def reconnect(self):
        if not self.connected:
            wlan = network.WLAN(network.STA_IF)
            if wlan.isconnected():
                self._connect()
    
    def disconnect(self):
        if self.client and self.connected:
            try:
                self.client.disconnect()
                sys.stdout.write("[MQTT] Disconnected\n")
            except:
                pass
            self.connected = False
    
    def get_stats(self):
        return {
            'connected': self.connected,
            'messages_sent': self.messages_sent,
            'connection_attempts': self.connection_attempts,
            'broker': config.MQTT_BROKER if config.MQTT_ENABLED else 'disabled'
        }