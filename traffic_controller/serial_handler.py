import time
import sys
import select
import config


class SerialHandler:
    
    def __init__(self):
        self.last_vehicle_count = 0
        self.last_time = -1
        self.last_density = "low"
        self.last_update_time = 0
        
        self.messages_received = 0
        
        sys.stdout.write("[Serial] USB Serial initialized\n")
        sys.stdout.write(f"[Serial] Baudrate: {config.SERIAL_BAUDRATE}\n")
        sys.stdout.write("[Serial] Waiting for AI connection...\n")
    
    def check_messages(self):
        try:
            if select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                
                if line:
                    line = line.strip()
                    
                    try:
                        parts = line.split(',')
                        if len(parts) == 3:
                            count = int(parts[0])
                            counting_time = int(parts[1])
                            density = parts[2].strip()
                            
                            self.messages_received += 1
                            self.last_update_time = time.time()
                            self.last_vehicle_count = count
                            self.last_time = counting_time
                            self.last_density = density
                            
                            if config.DEBUG:
                                msg = f"[Serial] RX: count={count}, time={counting_time}, density={density}\n"
                                sys.stdout.write(msg)
                            
                            return {
                                'count': count,
                                'time': counting_time,
                                'density': density
                            }
                    except ValueError:
                        if config.DEBUG:
                            sys.stdout.write(f"[Serial] Parse error: {line}\n")
                    except Exception as e:
                        if config.DEBUG:
                            sys.stdout.write(f"[Serial] Error: {e}\n")
            
            return None
            
        except Exception as e:
            return None
    
    def get_vehicle_count(self):
        return self.last_vehicle_count
    
    def get_time(self):
        return self.last_time
    
    def get_density(self):
        return self.last_density
    
    def get_stats(self):
        return {
            'rx': self.messages_received,
            'count': self.last_vehicle_count,
            'time': self.last_time,
            'density': self.last_density
        }