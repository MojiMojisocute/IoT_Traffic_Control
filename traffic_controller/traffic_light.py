import time
import config
from led_controller import LEDController


class TrafficLightController:
    STATE_RED = "RED"
    STATE_YELLOW = "YELLOW"
    STATE_GREEN = "GREEN"
    
    def __init__(self, led_controller):
        self.led = led_controller
        self.current_state = self.STATE_RED
        self.state_start_time = time.time()
        self.current_duration = config.RED_LIGHT_NORMAL
        
        self.auto_mode = True
        self.manual_override = False
        
        self.current_density = "low"
        self.vehicle_count = 0
        
        self.cycle_count = 0
        self.total_red_time = 0
        self.total_yellow_time = 0
        self.total_green_time = 0

        self.led.red_on()
        
        if config.DEBUG:
            print("[Traffic] Controller initialized")
            print(f"[Traffic] Starting with RED light ({self.current_duration}s)")
    
    def update(self):
        if self.manual_override:
            return
        
        elapsed = time.time() - self.state_start_time
        
        if elapsed >= self.current_duration:
            self._next_state()
    
    def _next_state(self):
        if self.current_state == self.STATE_RED:
            self.current_state = self.STATE_GREEN
            self.current_duration = config.GREEN_LIGHT_TIME
            self.led.green_on()
            self.total_red_time += time.time() - self.state_start_time
            
        elif self.current_state == self.STATE_GREEN:
            self.current_state = self.STATE_YELLOW
            self.current_duration = config.YELLOW_LIGHT_TIME
            self.led.yellow_on()
            self.total_green_time += time.time() - self.state_start_time
            
        elif self.current_state == self.STATE_YELLOW:
            self.current_state = self.STATE_RED
            self.current_duration = self._calculate_red_duration()
            self.led.red_on()
            self.total_yellow_time += time.time() - self.state_start_time
            self.cycle_count += 1
            
            if config.DEBUG:
                print(f"[Traffic] Cycle {self.cycle_count} complete")
        
        self.state_start_time = time.time()
        
        if config.DEBUG:
            print(f"[Traffic] State changed to {self.current_state} ({self.current_duration}s)")
    
    def _calculate_red_duration(self):
        if not self.auto_mode:
            return config.RED_LIGHT_NORMAL
        
        if self.current_density == "high":
            duration = config.RED_LIGHT_BUSY
            if config.DEBUG:
                print(f"[Traffic] HIGH density detected -> RED light {duration}s")
        
        elif self.current_density == "medium":
            duration = config.RED_LIGHT_MEDIUM
            if config.DEBUG:
                print(f"[Traffic] MEDIUM density detected -> RED light {duration}s")
        
        else:
            duration = config.RED_LIGHT_NORMAL
            if config.DEBUG:
                print(f"[Traffic] LOW density -> RED light {duration}s (normal)")
        
        return duration
    
    def update_traffic_data(self, vehicle_count, density):
        self.vehicle_count = vehicle_count
        self.current_density = density
        
        if config.DEBUG:
            print(f"[Traffic] Updated traffic data: {vehicle_count} vehicles, density={density}")
    
    def force_state(self, state, duration=None):
        self.manual_override = True
        self.current_state = state
        self.state_start_time = time.time()
        
        if duration:
            self.current_duration = duration
        else:
            if state == self.STATE_RED:
                self.current_duration = config.RED_LIGHT_NORMAL
            elif state == self.STATE_YELLOW:
                self.current_duration = config.YELLOW_LIGHT_TIME
            else:
                self.current_duration = config.GREEN_LIGHT_TIME
        
        if state == self.STATE_RED:
            self.led.red_on()
        elif state == self.STATE_YELLOW:
            self.led.yellow_on()
        else:
            self.led.green_on()
        
        if config.DEBUG:
            print(f"[Traffic] Manual override: {state} for {self.current_duration}s")
    
    def resume_auto(self):
        self.manual_override = False
        self.auto_mode = True
        if config.DEBUG:
            print("[Traffic] Resumed auto mode")
    
    def get_state(self):
        elapsed = time.time() - self.state_start_time
        remaining = max(0, self.current_duration - elapsed)
        
        return {
            'state': self.current_state,
            'elapsed': round(elapsed, 1),
            'remaining': round(remaining, 1),
            'duration': self.current_duration,
            'auto_mode': self.auto_mode,
            'manual_override': self.manual_override,
            'vehicle_count': self.vehicle_count,
            'density': self.current_density
        }
    
    def get_stats(self):
        return {
            'cycle_count': self.cycle_count,
            'total_red_time': round(self.total_red_time, 1),
            'total_yellow_time': round(self.total_yellow_time, 1),
            'total_green_time': round(self.total_green_time, 1),
            'current_state': self.current_state,
            'auto_mode': self.auto_mode
        }

    def reset_stats(self):
        self.cycle_count = 0
        self.total_red_time = 0
        self.total_yellow_time = 0
        self.total_green_time = 0
        if config.DEBUG:
            print("[Traffic] Statistics reset")