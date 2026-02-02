import time
import sys
import config


class TrafficLightController:
    
    STATE_RED = "RED"
    STATE_YELLOW = "YELLOW"
    STATE_GREEN = "GREEN"
    
    def __init__(self, led_controller):
        self.led = led_controller
        self.current_state = self.STATE_RED
        self.state_start_time = time.time()
        self.current_duration = config.RED_LIGHT_MAX
        
        self.current_density = "low"
        self.vehicle_count = 0
        self.previous_vehicle_count = 0
        
        self.cycle_count = 0
        self.total_red_time = 0
        self.total_yellow_time = 0
        self.total_green_time = 0
        self.time_adjustments = 0
        
        if config.DEBUG:
            sys.stdout.write("[Traffic] Controller initialized\n")
    
    def update(self):
        elapsed = time.time() - self.state_start_time
        
        if self.current_state == self.STATE_RED:
            if self._should_reduce_red_time():
                self._next_state()
                return
        
        elif self.current_state == self.STATE_GREEN:
            if self._should_switch_to_yellow():
                self._next_state()
                return
        
        if elapsed >= self.current_duration:
            self._next_state()
    
    def _should_reduce_red_time(self):
        elapsed = time.time() - self.state_start_time
        
        if self.vehicle_count >= config.THRESHOLD_CRITICAL:
            if elapsed >= config.RED_LIGHT_MIN:
                self.time_adjustments += 1
                if config.DEBUG:
                    sys.stdout.write(f"[Traffic] CRITICAL: {self.vehicle_count} vehicles - Switching to GREEN NOW\n")
                return True
        
        if self.vehicle_count >= config.THRESHOLD_HIGH:
            if self.previous_vehicle_count > 0 and self.vehicle_count > self.previous_vehicle_count:
                vehicle_increase = self.vehicle_count - self.previous_vehicle_count
                time_reduction = vehicle_increase * config.TIME_REDUCTION_PER_VEHICLE
                
                new_duration = max(config.RED_LIGHT_MIN, self.current_duration - time_reduction)
                
                if new_duration < self.current_duration:
                    self.current_duration = new_duration
                    self.time_adjustments += 1
                    
                    if config.DEBUG:
                        sys.stdout.write(f"[Traffic] HIGH TRAFFIC: {self.vehicle_count} vehicles (+{vehicle_increase})\n")
                        sys.stdout.write(f"[Traffic] Reduced RED time to {new_duration}s\n")
                
                if elapsed >= new_duration:
                    return True
        
        return False
    
    def _should_switch_to_yellow(self):
        elapsed = time.time() - self.state_start_time
        
        if self.vehicle_count < config.THRESHOLD_LOW:
            if elapsed >= config.GREEN_LIGHT_MIN:
                self.time_adjustments += 1
                
                if config.DEBUG:
                    sys.stdout.write(f"[Traffic] LOW TRAFFIC: {self.vehicle_count} vehicles - Ending GREEN early\n")
                
                return True
        
        return False
    
    def _next_state(self):
        if self.current_state == self.STATE_RED:
            self.current_state = self.STATE_GREEN
            self.current_duration = self._calculate_green_duration()
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
                sys.stdout.write(f"[Traffic] Cycle {self.cycle_count} completed\n")
        
        self.state_start_time = time.time()
        
        if config.DEBUG:
            sys.stdout.write(f"[Traffic] State: {self.current_state} | Duration: {self.current_duration}s | Vehicles: {self.vehicle_count}\n")
    
    def _calculate_red_duration(self):
        if self.vehicle_count >= config.THRESHOLD_CRITICAL:
            duration = config.RED_LIGHT_MIN
        elif self.vehicle_count >= config.THRESHOLD_HIGH:
            reduction = (self.vehicle_count - config.THRESHOLD_HIGH) * config.TIME_REDUCTION_PER_VEHICLE
            duration = config.RED_LIGHT_MAX - reduction
        else:
            duration = config.RED_LIGHT_MAX
        
        return max(config.RED_LIGHT_MIN, min(config.RED_LIGHT_MAX, duration))
    
    def _calculate_green_duration(self):
        if self.vehicle_count >= config.THRESHOLD_CRITICAL:
            duration = config.GREEN_LIGHT_MAX
        elif self.vehicle_count >= config.THRESHOLD_HIGH:
            duration = int(config.GREEN_LIGHT_MIN + (config.GREEN_LIGHT_MAX - config.GREEN_LIGHT_MIN) * 0.7)
        elif self.vehicle_count >= config.THRESHOLD_LOW:
            duration = int(config.GREEN_LIGHT_MIN + (config.GREEN_LIGHT_MAX - config.GREEN_LIGHT_MIN) * 0.4)
        else:
            duration = config.GREEN_LIGHT_MIN
        
        return max(config.GREEN_LIGHT_MIN, min(config.GREEN_LIGHT_MAX, duration))
    
    def update_traffic_data(self, vehicle_count, density):
        self.previous_vehicle_count = self.vehicle_count
        self.vehicle_count = vehicle_count
        self.current_density = density
        
        if config.DEBUG and self.vehicle_count != self.previous_vehicle_count:
            change = self.vehicle_count - self.previous_vehicle_count
            symbol = "UP" if change > 0 else "DOWN" if change < 0 else "SAME"
            sys.stdout.write(f"[Traffic] {symbol} Updated: {self.previous_vehicle_count} -> {self.vehicle_count} vehicles ({density})\n")
    
    def get_countdown(self):
        elapsed = time.time() - self.state_start_time
        remaining = int(self.current_duration - elapsed)
        return max(0, remaining)
    
    def get_light_state(self):
        if self.current_state == self.STATE_RED:
            return "red"
        elif self.current_state == self.STATE_YELLOW:
            return "yellow"
        elif self.current_state == self.STATE_GREEN:
            return "green"
        return "unknown"
    
    def get_stats(self):
        return {
            'cycle_count': self.cycle_count,
            'total_red_time': round(self.total_red_time, 1),
            'total_yellow_time': round(self.total_yellow_time, 1),
            'total_green_time': round(self.total_green_time, 1),
            'time_adjustments': self.time_adjustments,
            'current_state': self.current_state
        }