import time
import gc
from machine import Pin
import config
from utils import format_uptime, print_header, print_dict, Timer
from led_controller import LEDController
from serial_handler import SerialHandler
from traffic_light import TrafficLightController


class SmartTrafficSystem:
    def __init__(self):
        self.start_time = time.time()
        self.running = True
        
        self.led = None
        self.serial = None
        self.traffic = None
        
        if config.ENABLE_MANUAL_MODE:
            self.button = Pin(config.MANUAL_BUTTON_PIN, Pin.IN, Pin.PULL_UP)
            self.last_button_state = 1
        
        self.status_timer = Timer(10)
        self.serial_check_timer = Timer(config.UPDATE_INTERVAL)
        self.watchdog_timer = Timer(5)
        
        self.loop_count = 0
        self.errors = 0
        
        print_header("ESP32 SMART TRAFFIC LIGHT SYSTEM - SERIAL MODE")
        print("\nInitializing system...")
    
    def initialize(self):
        print("\n[1/3] Initializing LED Controller...")
        try:
            self.led = LEDController()
            self.led.test_sequence()
        except Exception as e:
            print(f"Failed to initialize LED: {e}")
            return False
        
        print("\n[2/3] Initializing Traffic Light Controller...")
        try:
            self.traffic = TrafficLightController(self.led)
        except Exception as e:
            print(f"Failed to initialize Traffic Controller: {e}")
            return False
        
        print("\n[3/3] Connecting to Serial...")
        try:
            self.serial = SerialHandler(on_message_callback=self._on_serial_message)
            
            if not self.serial.connect():
                print("⚠ Serial connection failed - running in offline mode")
            
        except Exception as e:
            print(f"⚠ Serial error: {e} - running in offline mode")
            self.serial = None
        
        print_header("SYSTEM READY")
        self._print_status()
        
        return True
    
    def _on_serial_message(self, data):
        try:
            if 'density' in data:
                density = data.get('density', 'low')
                count = data.get('count', 0)
                self.traffic.update_traffic_data(count, density)
            
            elif 'count' in data:
                count = data.get('count', 0)
                if count <= config.THRESHOLD_LOW:
                    density = 'low'
                elif count <= config.THRESHOLD_MEDIUM:
                    density = 'medium'
                else:
                    density = 'high'
                
                self.traffic.update_traffic_data(count, density)
        
        except Exception as e:
            print(f"[Main] Error processing serial message: {e}")
    
    def run(self):
        print("\nStarting main loop...")
        print("Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                self.loop_count += 1
                
                if self.serial and self.serial_check_timer.check():
                    if not self.serial.connected:
                        print("[Main] Serial disconnected - reconnecting...")
                        self.serial.connect()
                    else:
                        self.serial.check_messages()
                
                self.traffic.update()
                
                if self.serial and self.watchdog_timer.check():
                    if not self.serial.is_data_fresh():
                        if self.traffic.current_density != "low":
                            print("[Main] ⚠ No fresh data - reverting to normal mode")
                            self.traffic.update_traffic_data(0, "low")
                
                if config.ENABLE_MANUAL_MODE:
                    self._check_manual_button()
                
                if self.status_timer.check():
                    self._print_status()
                    
                    if self.serial and self.serial.connected:
                        state_info = self.traffic.get_state()
                        self.serial.send_data(state_info)
                
                if self.loop_count % 100 == 0:
                    gc.collect()
                
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            print("\n\nShutdown requested...")
            self.shutdown()
        
        except Exception as e:
            print(f"\n\nFatal error: {e}")
            import sys
            sys.print_exception(e)
            self.errors += 1
            self.shutdown()
    
    def _check_manual_button(self):
        current_state = self.button.value()
        
        if current_state == 0 and self.last_button_state == 1:
            print("[Main] Manual button pressed")
            
            if self.traffic.current_state == TrafficLightController.STATE_RED:
                self.traffic.force_state(TrafficLightController.STATE_GREEN)
            else:
                self.traffic.force_state(TrafficLightController.STATE_RED)
            
            time.sleep(0.5)
        
        self.last_button_state = current_state
    
    def _print_status(self):
        uptime = time.time() - self.start_time
        state = self.traffic.get_state()
        
        print("\n" + "-" * 60)
        print(f"Uptime: {format_uptime(uptime)} | Loops: {self.loop_count}")
        print(f"Light: {state['state']} | Time: {state['elapsed']:.1f}/{state['duration']}s")
        print(f"Vehicles: {state['vehicle_count']} | Density: {state['density']}")
        
        if self.serial:
            print(f"Serial: {'Connected' if self.serial.connected else 'Disconnected'} | " +
                  f"Messages: {self.serial.messages_received}")
        
        print("-" * 60)
    
    def shutdown(self):
        print("\nShutting down system...")
        
        self.running = False
        
        if self.serial and self.serial.connected:
            self.serial.send_data({'status': 'offline', 'timestamp': int(time.time())})
            time.sleep(0.5)
            self.serial.disconnect()
        
        if self.led:
            self.led.all_off()
        
        print_header("FINAL STATISTICS")
        
        print("\nTraffic Light:")
        print_dict(self.traffic.get_stats())
        
        if self.serial:
            print("\nSerial:")
            print_dict(self.serial.get_stats())
        
        print(f"\nTotal Loops: {self.loop_count}")
        print(f"Total Errors: {self.errors}")
        print(f"Uptime: {format_uptime(time.time() - self.start_time)}")
        
        print_header("SYSTEM STOPPED")


def main():
    try:
        system = SmartTrafficSystem()
        
        if system.initialize():
            system.run()
        else:
            print("\nSystem initialization failed!")
            if system.led:
                system.led.all_off()
    
    except Exception as e:
        print(f"\nCritical error: {e}")
        import sys
        sys.print_exception(e)


if __name__ == "__main__":
    main()