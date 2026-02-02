import time
import gc
import sys
import network

gc.collect()

print("\n" + "=" * 60)
print(" BOOT SEQUENCE")
print("=" * 60)

try:
    print("[BOOT] Connecting to WiFi...")
    import config
    
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if not wlan.isconnected():
        wlan.connect(config.WIFI_SSID, config.WIFI_PASSWORD)
        
        timeout = 15
        while not wlan.isconnected() and timeout > 0:
            print(".", end="")
            time.sleep(1)
            timeout -= 1
        
        if wlan.isconnected():
            print("\n[BOOT] WiFi Connected!")
            print(f"[BOOT] IP Address: {wlan.ifconfig()[0]}")
        else:
            print("\n[BOOT] WiFi connection timeout")
            print("[BOOT] MQTT will be disabled")
    else:
        print("[BOOT] Already connected to WiFi")
        print(f"[BOOT] IP Address: {wlan.ifconfig()[0]}")

except ImportError:
    print("[BOOT] config.py not found")
    print("[BOOT] MQTT will be disabled")
except Exception as e:
    print(f"[BOOT] WiFi error: {e}")
    print("[BOOT] MQTT will be disabled")

try:
    print("[BOOT] Importing main module...")
    import main
    print("[BOOT] Main module loaded successfully")
except Exception as e:
    print("[BOOT] ERROR loading main:")
    print(f"  {type(e).__name__}: {e}")
    sys.print_exception(e)
    print("\n[BOOT] System halted - check errors above")
    print("=" * 60)