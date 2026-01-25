#!/bin/bash

PORT="/dev/ttyUSB0"
BAUD=115200

echo "Uploading files to ESP32..."

ampy --port $PORT put config.py
ampy --port $PORT put led_controller.py
ampy --port $PORT put serial_handler.py
ampy --port $PORT put traffic_light.py
ampy --port $PORT put utils.py
ampy --port $PORT put main.py

echo "✓ Upload complete!"
echo "Restarting ESP32..."

ampy --port $PORT -- import machine; machine.reset()

echo "✓ ESP32 restarted"