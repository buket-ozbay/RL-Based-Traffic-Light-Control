# RL Based Traffic Light Control

# RL Based Traffic Light Control 🚦 (OpenCV + YOLO + Q-Learning + MQTT + Pico 2 W)

This project includes video-based vehicle detection (YOLOv5), queue density analysis and Q-Learning based traffic light control.

In addition, traffic queuing data is sent to a **Raspberry Pi Pico 2W** device using **the MQTT protocol**.

## 📂 Project Structure

```dart
├── traffic_light_control_RL_OpenCV.py   # RL environment, YOLO detection, Q-Learning agent, visualization
├── mqtt_publisher.py                    # Publishes queue data from RL environment to MQTT broker
├── mqtt_client_pico.py                  # MQTT client running on Pico 2W (MicroPython)
```

## 🚀 Installation

### 1. Python Environment

Make sure you have the following dependencies installed on your computer.

```
pip install opencv-python ultralytics paho-mqtt matplotlib numpy
```

If you have a CUDA supported video card, YOLO inference will run on **the GPU**, otherwise on the CPU.

### 2. MQTT Broker

The project defaults to a broker on `localhost` (127.0.0.1).

If you want to run your own broker:

```bash
# Mosquitto broker installation (Ubuntu)
sudo apt install mosquitto mosquitto-clients
sudo systemctl enable mosquitto

# For Windows: https://mosquitto.org/download/
```

- into`mosquitto.conf`:
    
    ```
    listener 1883 0.0.0.0
    allow_anonymous True
    ```
    
    add. You can also add username and password if you don't want to allow it.
      Note: If the MQTT Client is unable to connect, try turning off the firewall.
    
- Test (subscriber and publisher run on different terminals):
    
    ```powershell
    mosquitto_sub -h 127.0.0.1 -t test -v
    mosquitto_pub -h 127.0.0.1 -t test -m "hello"
    
    mosquitto_sub -h 192.168.1.1 -t test -v
    mosquitto_pub -h 192.168.1.1 -t test -m "hello"
    ```
    

### 3. Raspberry Pi Pico W (MicroPython)

- Flash Pico with **MicroPython** firmware.
- Load the`mqtt_client_pico.py` file into Pico.
- Fill in the **WiFi SSID / Password** fields at the beginning of the file:
    
    ```python
    WIFI_SSID     = "your_wifi"
    WIFI_PASSWORD = "your_pass"
    MQTT_BROKER   = "192.168.1.1"  # PC or broker IP address
    ```
    

---

## ⚙️ Operation Steps

### Step 1: RL Environment + MQTT Publisher

Run the traffic video and the RL agent on the computer:

```bash
python mqtt_publisher.py
```

- This file uses the RL environment in `traffic_light_control_RL_OpenCV.py`.
- It detects vehicles with YOLOv5, calculates queues.
- At each step, it sends messages **in JSON format** as follows:

```json
{"N": 3, "S": 2, "E": 5, "W": 1, "step": 42}
```

These messages are broadcast over MQTT to the `traffic/counts` topical.

---

### Step 2: Pico W MQTT Client

When `mqtt_client_pico.py` runs on Pico:

- Connects to the WiFi network.
- Subscribes to`traffic/counts` topical.
- Writes incoming messages to the screen via USB serial port.
- The LED flashes.
- Optionally sends summary information to the `traffic/status` community.

Sample output (in console):

```
[counts] step=42  N=3  S=2  E=5  W=1
```

---

### Step 3: Visualization (Optional)

The file`traffic_light_control_RL_OpenCV.py` can be run independently:

```bash
python traffic_light_control_RL_OpenCV.py
```

- Traffic lights, vehicles and actions of the RL agent are visualized with matplotlib.
- On the screen
    - 🚗 Vehicle ID and colors
    - 🚦 Light condition
    - 📊 The agent's reward and epsilon value are shown.

---

## 🧠 Methods Used

1. **YOLOv5 (Ultralytics)** → Vehicle detection (car, bus, truck, motorcycle).
    - YOLOv5 weights are subject to Ultralytics license.
2. **Vehicle Tracking** → ID based tracking, distribution to inbound/outbound queues.
3. **Q-Learning** →
    - State: Vertical road (NS) density, horizontal road (EW) density, light condition, duration bin
    - Action: [0] Continue, [1] Change phase
    - Reward: Waiting vehicle reduction, phase change penalty, small flow reward.
4. **MQTT** → Sending traffic densities from RL environment to Pico.
5. **Pico Client** → Real-time message reception and LED/serial port output on hardware.

---

## 📊 Sample Message Flow

```
[RL Publisher]  →  {"N":5,"S":3,"E":7,"W":2,"step":101}
                 ↓
[MQTT Broker]   →  topic: traffic/counts
                 ↓
[Pico Client]   →  [counts] step=101  N=5  S=3  E=7  W=2
```

---

## 👤 Author

***Buket ÖZBAY - Istanbul Technical University***

This project is prepared for traffic density control and IoT integration.

It is open to all kinds of contributions and suggestions for improvement. 🙌

For contact [ozbayb21@itu.edu.tr](mailto:ozbayb21@itu.edu.tr)
