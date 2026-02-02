import time
import gc
import sys
import network
import config
from led_controller import LEDController
from serial_handler import SerialHandler
from traffic_light import TrafficLightController
from mqtt_handler import MQTTHandler


sys.stdout.write("\n" + "=" * 60 + "\n")
sys.stdout.write(" ESP32 SMART TRAFFIC LIGHT CONTROLLER\n")
sys.stdout.write("=" * 60 + "\n")
sys.stdout.write("Firmware Version: 2.2\n")
sys.stdout.write("=" * 60 + "\n\n")

sys.stdout.write("[MAIN] Checking WiFi...\n")
wlan = network.WLAN(network.STA_IF)
wlan.active(True)

if not wlan.isconnected():
    sys.stdout.write(f"[MAIN] Connecting to {config.WIFI_SSID}...\n")
    wlan.disconnect()
    time.sleep(0.5)
    wlan.connect(config.WIFI_SSID, config.WIFI_PASSWORD)
    
    timeout = 15
    while not wlan.isconnected() and timeout > 0:
        sys.stdout.write(".")
        time.sleep(1)
        timeout -= 1
    
    sys.stdout.write("\n")

if wlan.isconnected():
    sys.stdout.write("[MAIN] WiFi Connected!\n")
    sys.stdout.write(f"[MAIN] IP: {wlan.ifconfig()[0]}\n")
    sys.stdout.write(f"[MAIN] Gateway: {wlan.ifconfig()[2]}\n\n")
else:
    sys.stdout.write("[MAIN] WiFi Failed - MQTT Disabled\n\n")

led = LEDController()
traffic = TrafficLightController(led)
serial = SerialHandler()
mqtt = MQTTHandler()

sys.stdout.write("[SYSTEM] All components initialized\n")
sys.stdout.write("[SYSTEM] Waiting for AI connection...\n")
sys.stdout.write("[SYSTEM] Status: BLINKING ALL LIGHTS\n\n")

blink_state = False
ai_started = False
last_blink = time.time()
last_check = time.time()
total_cycles = 0
start_time = time.time()

led.all_off()

try:
    while True:
        now = time.time()
        
        if now - last_check >= 0.01:
            data = serial.check_messages()
            
            if data:
                counting_time = data.get('time', -1)
                
                if counting_time >= 1 and not ai_started:
                    sys.stdout.write("\n" + "=" * 60 + "\n")
                    sys.stdout.write(" AI STARTED - BEGIN TRAFFIC CONTROL\n")
                    sys.stdout.write("=" * 60 + "\n\n")
                    led.all_off()
                    ai_started = True
                    traffic.state_start_time = now
                    traffic.led.red_on()
                    sys.stdout.write("[Traffic] Starting with RED light\n\n")
                
                if ai_started:
                    count = data.get('count', 0)
                    density = data.get('density', 'low')
                    traffic.update_traffic_data(count, density)
            
            if ai_started and (now - serial.last_update_time) > config.WATCHDOG_TIMEOUT:
                if traffic.current_state == traffic.STATE_RED:
                    sys.stdout.write("\n" + "=" * 60 + "\n")
                    sys.stdout.write(" WATCHDOG TIMEOUT - AI CONNECTION LOST\n")
                    sys.stdout.write(" STATE: RED - RETURNING TO BLINK MODE\n")
                    sys.stdout.write("=" * 60 + "\n\n")
                    ai_started = False
                    led.all_off()
                    blink_state = False
                    last_blink = now
                elif traffic.current_state == traffic.STATE_GREEN:
                    if not hasattr(traffic, 'waiting_for_cycle'):
                        sys.stdout.write("\n" + "=" * 60 + "\n")
                        sys.stdout.write(" WATCHDOG TIMEOUT - AI CONNECTION LOST\n")
                        sys.stdout.write(" STATE: GREEN - FINISHING CYCLE FIRST\n")
                        sys.stdout.write("=" * 60 + "\n\n")
                        traffic.waiting_for_cycle = True
                elif traffic.current_state == traffic.STATE_YELLOW:
                    if not hasattr(traffic, 'waiting_for_cycle'):
                        sys.stdout.write("\n" + "=" * 60 + "\n")
                        sys.stdout.write(" WATCHDOG TIMEOUT - AI CONNECTION LOST\n")
                        sys.stdout.write(" STATE: YELLOW - FINISHING CYCLE FIRST\n")
                        sys.stdout.write("=" * 60 + "\n\n")
                        traffic.waiting_for_cycle = True
            
            last_check = now
        
        if ai_started:
            traffic.update()
            
            countdown = traffic.get_countdown()
            light_state = traffic.get_light_state()
            vehicle_count = serial.get_vehicle_count()
            density = serial.get_density()
            
            mqtt.publish_status(light_state, countdown, vehicle_count, density)
            
            if hasattr(traffic, 'waiting_for_cycle') and traffic.current_state == traffic.STATE_RED:
                sys.stdout.write("\n[SYSTEM] Cycle completed - RETURNING TO BLINK MODE\n\n")
                ai_started = False
                delattr(traffic, 'waiting_for_cycle')
                led.all_off()
                blink_state = False
                last_blink = now
        else:
            mqtt.publish_status("blink", 0, 0, "none")
            
            if now - last_blink >= config.BLINK_INTERVAL:
                blink_state = not blink_state
                if blink_state:
                    led.all_on()
                else:
                    led.all_off()
                last_blink = now
        
        time.sleep(0.01)
        
        if ai_started and traffic.cycle_count > total_cycles:
            total_cycles = traffic.cycle_count
            if total_cycles % 10 == 0:
                gc.collect()

except KeyboardInterrupt:
    sys.stdout.write("\n" + "=" * 60 + "\n")
    sys.stdout.write(" SYSTEM STOPPED\n")
    sys.stdout.write("=" * 60 + "\n\n")
    
    stats = traffic.get_stats()
    serial_stats = serial.get_stats()
    mqtt_stats = mqtt.get_stats()
    uptime = time.time() - start_time
    
    sys.stdout.write("[Traffic Stats]\n")
    sys.stdout.write(f"  Total Cycles: {stats['cycle_count']}\n")
    sys.stdout.write(f"  Total Red Time: {stats['total_red_time']}s\n")
    sys.stdout.write(f"  Total Yellow Time: {stats['total_yellow_time']}s\n")
    sys.stdout.write(f"  Total Green Time: {stats['total_green_time']}s\n")
    sys.stdout.write(f"  Time Adjustments: {stats['time_adjustments']}\n")
    sys.stdout.write("\n[Serial Stats]\n")
    sys.stdout.write(f"  Messages Received: {serial_stats['rx']}\n")
    sys.stdout.write("\n[MQTT Stats]\n")
    sys.stdout.write(f"  Connected: {mqtt_stats['connected']}\n")
    sys.stdout.write(f"  Messages Sent: {mqtt_stats['messages_sent']}\n")
    sys.stdout.write(f"  Broker: {mqtt_stats['broker']}\n")
    sys.stdout.write(f"\n[System] Uptime: {uptime:.0f}s\n\n")
    
    mqtt.disconnect()
    led.all_off()
    sys.stdout.write("[SYSTEM] Cleanup complete\n\n")

except Exception as e:
    sys.stdout.write(f"\n[ERROR] {e}\n\n")
    mqtt.disconnect()
    led.all_off()