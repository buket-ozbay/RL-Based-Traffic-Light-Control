import time
import json
import paho.mqtt.client as mqtt

# RL module
from traffic_light_control_RL_OpenCV import TrafficEnvFromVideo, QLearningAgent

MQTT_BROKER   = "127.0.0.1"   # broker IP/hostname
MQTT_PORT     = 1883
MQTT_USER     = None
MQTT_PASS     = None
TOPIC_COUNTS  = "traffic/counts"  # inbound queue numbers will be sent to this topic

# MQTT
class MqttPublisher:
    def __init__(self, host, port=1883, user=None, password=None):
        self.c = mqtt.Client(client_id="rl-host-counts", callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        if user or password:
            self.c.username_pw_set(user, password)
        self.c.connect(host, port, keepalive=30)
        self.c.loop_start()

    def publish_counts(self, counts: dict):
        payload = json.dumps(counts)
        print("[MQTT] publish:", payload)
        self.c.publish(TOPIC_COUNTS, payload, qos=0, retain=False)

    def close(self):
        try:
            self.c.loop_stop()
            self.c.disconnect()
        except:
            pass

def run():
    # Video path on your PC
    VIDEO = r"C:\Users\Administrator\Downloads\camera_junction2.mp4"
    import cv2
    DEVICE = "cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    env = TrafficEnvFromVideo(VIDEO, show_debug=False, device=DEVICE)
    agent = QLearningAgent(n_actions=2, alpha=0.05, gamma=0.95,
                           epsilon=1.0, eps_min=0.05, eps_decay=0.995)

    bus = MqttPublisher(MQTT_BROKER, MQTT_PORT, MQTT_USER, MQTT_PASS)

    try:
        step = 0
        while True:
            s = env.get_state()
            a = agent.choose_action(s)
            s2, r = env.step(a)
            agent.learn(s, a, r, s2)
            agent.decay()
            step += 1

            # inbound queues in the directions
            counts = {
                "N": len(env.queues["N_in"]),
                "S": len(env.queues["S_in"]),
                "E": len(env.queues["E_in"]),
                "W": len(env.queues["W_in"]),
                "step": step
            }
            bus.publish_counts(counts)

            time.sleep(0.5)  # message frequency

    except KeyboardInterrupt:
        print("CTRL+C - exit.")
    finally:
        bus.close()

if __name__ == "__main__":
    run()
