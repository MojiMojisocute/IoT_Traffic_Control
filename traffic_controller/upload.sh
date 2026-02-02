#!/bin/bash

PORT="/dev/ttyUSB0"
BAUD=115200

echo "======================================================================"
echo "  ESP32 TRAFFIC CONTROLLER - FILE UPLOAD"
echo "======================================================================"
echo ""

if [ ! -c "$PORT" ]; then
    echo "ERROR: Port $PORT not found"
    echo ""
    echo "Available ports:"
    ls -l /dev/ttyUSB* 2>/dev/null || echo "  No USB serial ports found"
    echo ""
    exit 1
fi

echo "Port: $PORT"
echo "Baudrate: $BAUD"
echo ""
echo "----------------------------------------------------------------------"
echo "Uploading files to ESP32..."
echo "----------------------------------------------------------------------"
echo ""

FILES="boot.py config.py led_controller.py serial_handler.py traffic_light.py mqtt_handler.py main.py"

for file in $FILES; do
    if [ -f "$file" ]; then
        echo "Uploading: $file"
        
        ampy --port $PORT --baud $BAUD rm $file 2>/dev/null
        
        if [ "$file" == "main.py" ]; then
            if ampy --port $PORT --baud $BAUD put $file main.py; then
                echo "   Success"
            else
                echo "   Failed"
                exit 1
            fi
        else
            if ampy --port $PORT --baud $BAUD put $file; then
                echo "   Success"
            else
                echo "   Failed"
                exit 1
            fi
        fi
    else
        echo "Warning: $file not found, skipping..."
    fi
done

echo ""
echo "======================================================================"
echo "  UPLOAD COMPLETE"
echo "======================================================================"
echo ""
echo "Files uploaded. Press RESET button on ESP32 to start"
echo ""