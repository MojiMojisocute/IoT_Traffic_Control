"""
ESP32 Traffic Light System - Utilities
ฟังก์ชันเสริมต่างๆ
"""

import network
import time
import config


def connect_wifi(ssid=None, password=None, timeout=20):
    """
    เชื่อมต่อ WiFi
    
    Args:
        ssid: WiFi SSID (ถ้าไม่ระบุใช้จาก config)
        password: WiFi password (ถ้าไม่ระบุใช้จาก config)
        timeout: timeout ในการเชื่อมต่อ (วินาที)
    
    Returns:
        True ถ้าเชื่อมต่อสำเร็จ
    """
    ssid = ssid or config.WIFI_SSID
    password = password or config.WIFI_PASSWORD
    
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    
    if wlan.isconnected():
        print(f"[WiFi] Already connected to {ssid}")
        print(f"[WiFi] IP: {wlan.ifconfig()[0]}")
        return True
    
    print(f"[WiFi] Connecting to {ssid}...")
    wlan.connect(ssid, password)
    
    start_time = time.time()
    while not wlan.isconnected():
        if time.time() - start_time > timeout:
            print("[WiFi] ✗ Connection timeout")
            return False
        
        time.sleep(0.5)
        print(".", end="")
    
    print()
    print("[WiFi] ✓ Connected successfully")
    print(f"[WiFi] IP: {wlan.ifconfig()[0]}")
    print(f"[WiFi] Subnet: {wlan.ifconfig()[1]}")
    print(f"[WiFi] Gateway: {wlan.ifconfig()[2]}")
    
    return True


def check_wifi_connection():
    """
    ตรวจสอบว่าเชื่อมต่อ WiFi อยู่หรือไม่
    
    Returns:
        True ถ้าเชื่อมต่ออยู่
    """
    wlan = network.WLAN(network.STA_IF)
    return wlan.isconnected()


def get_wifi_info():
    """
    ดึงข้อมูล WiFi
    
    Returns:
        dict ของข้อมูล WiFi
    """
    wlan = network.WLAN(network.STA_IF)
    
    if not wlan.isconnected():
        return {'connected': False}
    
    config = wlan.ifconfig()
    return {
        'connected': True,
        'ip': config[0],
        'subnet': config[1],
        'gateway': config[2],
        'dns': config[3],
        'ssid': wlan.config('essid'),
        'rssi': wlan.status('rssi')
    }


def format_uptime(seconds):
    """
    แปลงวินาทีเป็นรูปแบบที่อ่านง่าย
    
    Args:
        seconds: จำนวนวินาที
    
    Returns:
        string เช่น "2d 5h 30m 15s"
    """
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    
    return " ".join(parts)


def print_header(title):
    """พิมพ์ header สวยๆ"""
    print()
    print("=" * 60)
    print(title.center(60))
    print("=" * 60)


def print_dict(data, indent=2):
    """
    พิมพ์ dictionary สวยๆ
    
    Args:
        data: dictionary ที่ต้องการพิมพ์
        indent: จำนวนช่องว่างหน้า
    """
    spaces = " " * indent
    for key, value in data.items():
        print(f"{spaces}{key}: {value}")


def blink_led(led, times=3, delay=0.2):
    """
    กระพริบ LED
    
    Args:
        led: Pin object
        times: จำนวนครั้ง
        delay: ความล่าช้าระหว่างกระพริบ (วินาที)
    """
    for _ in range(times):
        led.on()
        time.sleep(delay)
        led.off()
        time.sleep(delay)


def safe_divide(numerator, denominator, default=0):
    """
    หารอย่างปลอดภัย
    
    Args:
        numerator: ตัวตั้ง
        denominator: ตัวหาร
        default: ค่าที่จะคืนถ้าหารไม่ได้
    
    Returns:
        ผลหาร หรือ default ถ้าหารไม่ได้
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except:
        return default


def limit_value(value, min_val, max_val):
    """
    จำกัดค่าให้อยู่ในช่วง
    
    Args:
        value: ค่าที่ต้องการจำกัด
        min_val: ค่าต่ำสุด
        max_val: ค่าสูงสุด
    
    Returns:
        ค่าที่ถูกจำกัด
    """
    return max(min_val, min(max_val, value))


def map_range(value, in_min, in_max, out_min, out_max):
    """
    แปลงค่าจากช่วงหนึ่งไปอีกช่วงหนึ่ง (เหมือน Arduino map)
    
    Args:
        value: ค่าที่ต้องการแปลง
        in_min: ค่าต่ำสุดของช่วงต้นทาง
        in_max: ค่าสูงสุดของช่วงต้นทาง
        out_min: ค่าต่ำสุดของช่วงปลายทาง
        out_max: ค่าสูงสุดของช่วงปลายทาง
    
    Returns:
        ค่าที่แปลงแล้ว
    """
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min


class Timer:
    """ตัวจับเวลาอย่างง่าย"""
    
    def __init__(self, interval):
        """
        สร้างตัวจับเวลา
        
        Args:
            interval: ช่วงเวลา (วินาที)
        """
        self.interval = interval
        self.last_time = time.time()
    
    def check(self):
        """
        ตรวจสอบว่าถึงเวลาหรือยัง
        
        Returns:
            True ถ้าถึงเวลาแล้ว
        """
        current_time = time.time()
        if current_time - self.last_time >= self.interval:
            self.last_time = current_time
            return True
        return False
    
    def reset(self):
        """รีเซ็ตตัวจับเวลา"""
        self.last_time = time.time()
    
    def elapsed(self):
        """
        ดูว่าผ่านไปกี่วินาทีแล้ว
        
        Returns:
            จำนวนวินาทีที่ผ่านไป
        """
        return time.time() - self.last_time