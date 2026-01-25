import serial
import json
import time
import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SerialHandler:
    def __init__(self, on_message_callback=None):
        self.serial = None
        self.connected = False
        self.on_message_callback = on_message_callback
        
        self.last_vehicle_count = 0
        self.last_active_count = 0
        self.last_density = "low"
        self.last_update_time = 0
        
        self.messages_received = 0
        self.connection_attempts = 0
        
        if config.DEBUG:
            print("[Serial] Handler initialized")
    
    def connect(self):
        try:
            self.connection_attempts += 1
            
            if config.DEBUG:
                print(f"[Serial] Connecting to {config.SERIAL_PORT} at {config.SERIAL_BAUDRATE} baud")
            
            self.serial = serial.Serial(
                port=config.SERIAL_PORT,
                baudrate=config.SERIAL_BAUDRATE,
                timeout=1,
                write_timeout=1
            )
            
            self.connected = True
            
            if config.DEBUG:
                print("[Serial] ✓ Connected successfully")
            
            return True
            
        except Exception as e:
            self.connected = False
            print(f"[Serial] ✗ Connection failed: {e}")
            return False
    
    def disconnect(self):
        try:
            if self.serial and self.serial.is_open:
                self.serial.close()
            self.connected = False
            if config.DEBUG:
                print("[Serial] Disconnected")
        except Exception as e:
            print(f"[Serial] Error disconnecting: {e}")
    
    def send_data(self, data):
        if not self.connected or not self.serial:
            return False
        
        try:
            payload = json.dumps(data) + "\n"
            self.serial.write(payload.encode('utf-8'))
            
            if config.DEBUG:
                print(f"[Serial] Sent: {payload.strip()}")
            
            return True
            
        except Exception as e:
            print(f"[Serial] Error sending: {e}")
            return False
    
    def check_messages(self):
        if not self.connected or not self.serial:
            return False
        
        try:
            if self.serial.in_waiting > 0:
                line = self.serial.readline().decode('utf-8').strip()
                
                if line:
                    self.messages_received += 1
                    self.last_update_time = time.time()
                    
                    data = json.loads(line)
                    
                    if 'count' in data:
                        self.last_active_count = data.get('count', 0)
                    
                    if 'density' in data:
                        self.last_density = data.get('density', 'low')
                    
                    if self.on_message_callback:
                        self.on_message_callback(data)
                    
                    if config.DEBUG:
                        print(f"[Serial] Received: {line}")
            
            return True
        except Exception as e:
            print(f"[Serial] Error checking messages: {e}")
            return False
    
    def get_vehicle_count(self):
        return self.last_active_count
    
    def get_density(self):
        return self.last_density
    
    def is_data_fresh(self, timeout=None):
        if timeout is None:
            timeout = config.WATCHDOG_TIMEOUT
        
        if self.last_update_time == 0:
            return False
        
        elapsed = time.time() - self.last_update_time
        return elapsed < timeout
    
    def get_stats(self):
        return {
            'connected': self.connected,
            'messages_received': self.messages_received,
            'connection_attempts': self.connection_attempts,
            'last_active_count': self.last_active_count,
            'last_density': self.last_density,
            'data_fresh': self.is_data_fresh()
        }