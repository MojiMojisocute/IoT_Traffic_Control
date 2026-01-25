from machine import Pin
import config


class LEDController:
    """ควบคุมไฟ LED ไฟจราจร"""
    
    def __init__(self):
        """สร้าง pin สำหรับควบคุมไฟแต่ละสี"""
        self.red = Pin(config.PIN_RED_LIGHT, Pin.OUT)
        self.yellow = Pin(config.PIN_YELLOW_LIGHT, Pin.OUT)
        self.green = Pin(config.PIN_GREEN_LIGHT, Pin.OUT)
        
        # ปิดไฟทั้งหมดตอนเริ่มต้น
        self.all_off()
        
        if config.DEBUG:
            print("[LED] Controller initialized")
            print(f"  Red: GPIO{config.PIN_RED_LIGHT}")
            print(f"  Yellow: GPIO{config.PIN_YELLOW_LIGHT}")
            print(f"  Green: GPIO{config.PIN_GREEN_LIGHT}")
    
    def red_on(self):
        """เปิดไฟแดง ปิดไฟอื่น"""
        self.red.on()
        self.yellow.off()
        self.green.off()
        if config.DEBUG:
            print("[LED] 🔴 RED ON")
    
    def yellow_on(self):
        """เปิดไฟเหลือง ปิดไฟอื่น"""
        self.red.off()
        self.yellow.on()
        self.green.off()
        if config.DEBUG:
            print("[LED] 🟡 YELLOW ON")
    
    def green_on(self):
        """เปิดไฟเขียว ปิดไฟอื่น"""
        self.red.off()
        self.yellow.off()
        self.green.on()
        if config.DEBUG:
            print("[LED] 🟢 GREEN ON")
    
    def all_off(self):
        """ปิดไฟทั้งหมด"""
        self.red.off()
        self.yellow.off()
        self.green.off()
        if config.DEBUG:
            print("[LED] All lights OFF")
    
    def all_on(self):
        """เปิดไฟทั้งหมด (ทดสอบเท่านั้น)"""
        self.red.on()
        self.yellow.on()
        self.green.on()
    
    def test_sequence(self):
        """ทดสอบไฟทั้งหมด"""
        import time
        
        print("[LED] Testing sequence...")
        
        self.red_on()
        time.sleep(1)
        
        self.yellow_on()
        time.sleep(1)
        
        self.green_on()
        time.sleep(1)
        
        self.all_off()
        print("[LED] Test complete")
    
    def get_current_light(self):
        """ดูว่าไฟไหนติดอยู่"""
        if self.red.value():
            return "RED"
        elif self.yellow.value():
            return "YELLOW"
        elif self.green.value():
            return "GREEN"
        else:
            return "OFF"
    
    def cleanup(self):
        """ทำความสะอาดก่อนปิดโปรแกรม"""
        self.all_off()
        if config.DEBUG:
            print("[LED] Cleanup done")