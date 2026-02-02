const MQTT_BROKER = 'wss://broker.hivemq.com:8884/mqtt';
const MQTT_TOPIC = 'traffic/esp32/status';

const redLight = document.getElementById('redLight');
const yellowLight = document.getElementById('yellowLight');
const greenLight = document.getElementById('greenLight');
const countdown = document.getElementById('countdown');
const vehicles = document.getElementById('vehicles');
const density = document.getElementById('density');
const densityBar = document.getElementById('densityBar');
const currentLight = document.getElementById('currentLight');
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');

const client = mqtt.connect(MQTT_BROKER);

client.on('connect', () => {
    console.log('Connected to MQTT broker');
    statusDot.className = 'status-dot connected';
    statusText.textContent = 'Connected to ESP32';
    client.subscribe(MQTT_TOPIC);
});

client.on('error', (err) => {
    console.error('Connection error:', err);
    statusDot.className = 'status-dot disconnected';
    statusText.textContent = 'Connection Error';
});

client.on('message', (topic, message) => {
    try {
        const data = JSON.parse(message.toString());
        updateDisplay(data);
    } catch (e) {
        console.error('Parse error:', e);
    }
});

function updateDisplay(data) {
    redLight.classList.remove('active', 'blink');
    yellowLight.classList.remove('active', 'blink');
    greenLight.classList.remove('active', 'blink');

    if (data.light === 'red') {
        redLight.classList.add('active');
        currentLight.textContent = '🔴 Red';
    } else if (data.light === 'yellow') {
        yellowLight.classList.add('active');
        currentLight.textContent = '🟡 Yellow';
    } else if (data.light === 'green') {
        greenLight.classList.add('active');
        currentLight.textContent = '🟢 Green';
    } else if (data.light === 'blink') {
        redLight.classList.add('blink');
        yellowLight.classList.add('blink');
        greenLight.classList.add('blink');
        currentLight.textContent = '⚠️ Standby';
    }

    countdown.textContent = data.countdown || '--';

    vehicles.textContent = data.vehicles || 0;

    const densityText = data.density || 'none';
    if (densityText === 'low') {
        density.textContent = 'Low';
        densityBar.className = 'density-fill density-low';
    } else if (densityText === 'medium') {
        density.textContent = 'Medium';
        densityBar.className = 'density-fill density-medium';
    } else if (densityText === 'high') {
        density.textContent = 'High';
        densityBar.className = 'density-fill density-high';
    } else {
        density.textContent = 'None';
        densityBar.className = 'density-fill';
        densityBar.style.width = '0%';
    }
}

setInterval(() => {
    if (!client.connected) {
        statusDot.className = 'status-dot disconnected';
        statusText.textContent = 'Disconnected';
    }
}, 5000);