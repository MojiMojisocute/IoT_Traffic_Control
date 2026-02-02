import time
import gc
import sys
import network

gc.collect()

print("\n" + "=" * 60)
print(" BOOT SEQUENCE")
print("=" * 60)

wlan = None

try:
    print("[BOOT] Importing config...")
    import config
    
    print("[BOOT] Initializing WiFi...")
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if wlan.isconnected():
        print("[BOOT] Already connected")
        print(f"[BOOT] IP: {wlan.ifconfig()[0]}")
        print(f"[BOOT] Gateway: {wlan.ifconfig()[2]}")
    else:
        print(f"[BOOT] SSID: {config.WIFI_SSID}")
        print("[BOOT] Connecting...", end="")
        
        wlan.disconnect()
        time.sleep(1)
        wlan.connect(config.WIFI_SSID, config.WIFI_PASSWORD)
        
        timeout = 20
        while not wlan.isconnected() and timeout > 0:
            print(".", end="")
            time.sleep(1)
            timeout -= 1
        
        print()
        
        if wlan.isconnected():
            print("[BOOT] WiFi Connected!")
            print(f"[BOOT] IP: {wlan.ifconfig()[0]}")
            print(f"[BOOT] Netmask: {wlan.ifconfig()[1]}")
            print(f"[BOOT] Gateway: {wlan.ifconfig()[2]}")
            print(f"[BOOT] DNS: {wlan.ifconfig()[3]}")
        else:
            print("[BOOT] WiFi timeout")
            print("[BOOT] MQTT disabled")
            wlan.active(False)

except ImportError:
    print("[BOOT] config.py not found")
    print("[BOOT] MQTT disabled")
except Exception as e:
    print(f"[BOOT] WiFi error: {e}")
    print("[BOOT] MQTT disabled")
    if wlan:
        wlan.active(False)

print("=" * 60)
print()

try:
    import main
except Exception as e:
    print(f"[BOOT] Main error: {e}")
    sys.print_exception(e)