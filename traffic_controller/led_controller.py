from machine import Pin
import sys
import config


class LEDController:
    
    def __init__(self):
        self.red = Pin(config.PIN_RED_LIGHT, Pin.OUT)
        self.yellow = Pin(config.PIN_YELLOW_LIGHT, Pin.OUT)
        self.green = Pin(config.PIN_GREEN_LIGHT, Pin.OUT)
        
        self.all_on()
        
        if config.DEBUG:
            sys.stdout.write("[LED] Traffic Light initialized\n")
            sys.stdout.write(f"[LED] Red: GPIO{config.PIN_RED_LIGHT}\n")
            sys.stdout.write(f"[LED] Yellow: GPIO{config.PIN_YELLOW_LIGHT}\n")
            sys.stdout.write(f"[LED] Green: GPIO{config.PIN_GREEN_LIGHT}\n")
    
    def red_on(self):
        self.red.on()
        self.yellow.off()
        self.green.off()
        if config.DEBUG:
            sys.stdout.write("[LED] RED ON\n")
    
    def yellow_on(self):
        self.red.off()
        self.yellow.on()
        self.green.off()
        if config.DEBUG:
            sys.stdout.write("[LED] YELLOW ON\n")
    
    def green_on(self):
        self.red.off()
        self.yellow.off()
        self.green.on()
        if config.DEBUG:
            sys.stdout.write("[LED] GREEN ON\n")
    
    def all_off(self):
        self.red.off()
        self.yellow.off()
        self.green.off()
    
    def all_on(self):
        self.red.on()
        self.yellow.on()
        self.green.on()
    
    def cleanup(self):
        self.all_off()