# Host sent the topic "traffic/counts" as follows:
#   {"N":3,"S":2,"E":5,"W":1,"step":42}

import network, time, json, sys
from umqtt.simple import MQTTClient
from machine import Pin

# CONFIG
WIFI_SSID     = ""
WIFI_PASSWORD = ""

MQTT_BROKER   = "192.168.1.1"   # IP/hostname
MQTT_PORT     = 1883
MQTT_USER     = None              
MQTT_PASS     = None
CLIENT_ID     = "pico2w-counts-01"

TOPIC_COUNTS  = b"traffic/counts"   # Incoming data from RL
TOPIC_STATUS  = b"traffic/status"   # Optional feedback

STATUS_ECHO   = True  # If it is True, summary is published

# Onboard LED (For Pico W/Pico 2 W)
try:
    LED = Pin("LED", Pin.OUT, value=0)
except:
    LED = None

# WIFI
def wifi_connect():
    wlan = network.WLAN(network.STA_IF)
    if not wlan.active():
        wlan.active(True)
    if not wlan.isconnected():
        print("WiFi is connecting...")
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)
        t0 = time.time()
        while not wlan.isconnected():
            time.sleep(0.2)
            if time.time() - t0 > 20:
                raise RuntimeError("WiFi cannot be connected")
    print("WiFi OK:", wlan.ifconfig())
    return wlan

# MQTT
mqtt = None
last_counts = {}

def on_msg(topic, msg):
    global last_counts
    # for debug
    if LED:
        LED.on()
    try:
        data = json.loads(msg)
        # N,S,E,W in the intersection
        n = int(data.get("N", 0))
        s = int(data.get("S", 0))
        e = int(data.get("E", 0))
        w = int(data.get("W", 0))
        step = int(data.get("step", -1))

        last_counts = {"N": n, "S": s, "E": e, "W": w, "step": step}

        # Written to the console
        print(f"[counts] step={step}  N={n}  S={s}  E={e}  W={w}")

        # Summary if it is wanted
        if STATUS_ECHO and mqtt:
            payload = json.dumps({"ok": True, "step": step, "sum": n+s+e+w})
            mqtt.publish(TOPIC_STATUS, payload)

    except Exception as e:
        print("Mesaj parse error:", e, "msg=", msg)
        if STATUS_ECHO and mqtt:
            mqtt.publish(TOPIC_STATUS, json.dumps({"ok": False, "err": str(e)}))
    finally:
        if LED:
            time.sleep(0.05)
            LED.off()

def mqtt_connect():
    c = MQTTClient(CLIENT_ID, MQTT_BROKER, port=MQTT_PORT,
                   user=MQTT_USER, password=MQTT_PASS, keepalive=30)
    c.set_callback(on_msg)
    c.connect()
    c.subscribe(TOPIC_COUNTS)
    print("MQTT is connected, listening →", TOPIC_COUNTS)
    return c

def main():
    global mqtt
    try:
        wifi_connect()
    except Exception as e:
        print("WiFi error:", e)
        time.sleep(5)
        sys.exit()


    try:
        mqtt = mqtt_connect()
        last_status = time.time()
        while True:
            # Non-blocking control, if there is a message, callback is triggered
            mqtt.check_msg()
            # little sleep
            time.sleep(0.02)
    except Exception as e:
        print("MQTT connection error:", e)
        try:
            if mqtt:
                mqtt.disconnect()
        except:
            pass
        time.sleep(2)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        try:
            if mqtt:
                mqtt.disconnect()
        except:
            pass
        print("Exit")