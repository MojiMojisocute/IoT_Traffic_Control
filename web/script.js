const MQTT_BROKER = 'wss://broker.hivemq.com:8884/mqtt';
const MQTT_TOPIC = 'traffic/esp32/status';

// DOM Elements
const redLight = document.getElementById('redLight');
const yellowLight = document.getElementById('yellowLight');
const greenLight = document.getElementById('greenLight');
const countdown = document.getElementById('countdown');
const vehicles = document.getElementById('vehicles');
const density = document.getElementById('density');
const densityBar = document.getElementById('densityBar');
const vehicleBar = document.getElementById('vehicleBar');
const currentLight = document.getElementById('currentLight');
const statusBadge = document.getElementById('statusBadge');
const statusText = document.getElementById('statusText');

// Initialize MQTT Client
const client = mqtt.connect(MQTT_BROKER);

client.on('connect', () => {
    console.log('Connected to MQTT broker');
    client.subscribe(MQTT_TOPIC);
});

client.on('error', (err) => {
    console.error('Connection error:', err);
    updateConnectionStatus(false);
});

client.on('offline', () => {
    console.log('Client offline');
    updateConnectionStatus(false);
});

client.on('reconnect', () => {
    console.log('Reconnecting...');
});

client.on('message', (topic, message) => {
    try {
        const data = JSON.parse(message.toString());
        updateDisplay(data);
    } catch (e) {
        console.error('Parse error:', e);
    }
});

// Update connection status
function updateConnectionStatus(isConnected) {
    if (isConnected) {
        statusBadge.classList.remove('disconnected');
        statusText.textContent = 'Connected';
    } else {
        statusBadge.classList.add('disconnected');
        statusText.textContent = 'Disconnected';
    }
}

// Update display with new data
function updateDisplay(data) {
    // Clear all light states
    redLight.classList.remove('active', 'blink');
    yellowLight.classList.remove('active', 'blink');
    greenLight.classList.remove('active', 'blink');
    
    // Remove all status classes
    currentLight.classList.remove('red', 'yellow', 'standby');

    // Update light state and connection status based on light
    switch(data.light) {
        case 'red':
            redLight.classList.add('active');
            currentLight.textContent = 'Red Light';
            currentLight.classList.add('red');
            updateConnectionStatus(true);
            break;
        case 'yellow':
            yellowLight.classList.add('active');
            currentLight.textContent = 'Yellow Light';
            currentLight.classList.add('yellow');
            updateConnectionStatus(true);
            break;
        case 'green':
            greenLight.classList.add('active');
            currentLight.textContent = 'Green Light';
            updateConnectionStatus(true);
            break;
        case 'blink':
            redLight.classList.add('blink');
            yellowLight.classList.add('blink');
            greenLight.classList.add('blink');
            currentLight.textContent = 'Standby';
            currentLight.classList.add('standby');
            updateConnectionStatus(true);
            break;
        default:
            currentLight.textContent = 'Unknown';
            currentLight.classList.add('standby');
            updateConnectionStatus(false);
    }

    // Update countdown
    countdown.textContent = data.countdown || '--';

    // Update vehicle count
    const vehicleCount = data.vehicles || 0;
    vehicles.textContent = vehicleCount;
    
    // Update vehicle bar (max 30 vehicles = 100%)
    const vehiclePercent = Math.min((vehicleCount / 30) * 100, 100);
    vehicleBar.style.width = vehiclePercent + '%';

    // Update density
    const densityText = data.density || 'none';
    updateDensity(densityText);
}

// Update density display
function updateDensity(densityLevel) {
    switch(densityLevel) {
        case 'low':
            density.textContent = 'Low';
            densityBar.style.width = '33%';
            break;
        case 'medium':
            density.textContent = 'Medium';
            densityBar.style.width = '66%';
            break;
        case 'high':
            density.textContent = 'High';
            densityBar.style.width = '100%';
            break;
        default:
            density.textContent = 'None';
            densityBar.style.width = '0%';
    }
}

console.log('Traffic Report System initialized');
console.log('Connecting to:', MQTT_BROKER);