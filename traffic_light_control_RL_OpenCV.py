from __future__ import annotations
import os
import cv2
import math
import time
import random
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation


# Helper functions
def clamp_roi(x, y, w, h, W, H):
    x = max(0, min(int(x), W-1))
    y = max(0, min(int(y), H-1))
    w = max(1, min(int(w), W-x))
    h = max(1, min(int(h), H-y))
    return (x, y, w, h)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0: return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / max(1e-6, (a_area + b_area - inter))


# Eliminate repeated boxes (IoU based)
def dedup_detections(dets, iou_thr=0.7):
    """
    dets: list[(cx,cy,xyxy,cname,conf)]
    If repeated boxes belong to the same object, the one has (IoU >= iou_thr) the most conf will stay.
    """
    if not dets:
        return dets
    dets = sorted(dets, key=lambda d: d[4], reverse=True)
    keep = []
    for d in dets:
        drop = False
        for k in keep:
            if iou_xyxy(d[2], k[2]) >= iou_thr:
                drop = True
                break
        if not drop:
            keep.append(d)
    return keep

# Hybrid cost: distance + (1 - IoU)
def _mt_cost(det, trk, img_diag, w_iou=0.6, w_dist=0.4):
    cx, cy, xyxy, _, _ = det
    tcx, tcy = trk['pos']
    iou = iou_xyxy(xyxy, trk.get('xyxy', (tcx-5, tcy-5, tcx+5, tcy+5)))
    dist = math.hypot(cx - tcx, cy - tcy) / (img_diag + 1e-6)
    return w_dist*dist + w_iou*(1.0 - iou)


# Dehaze (optional / not worked with my video)

def dark_channel_dehaze(bgr, radius=7, omega=0.95, t0=0.1):
    """Basit ve hızlı dark-channel dehaze. İstersen kapatabilirsin."""
    # BGR -> RGB
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2*radius+1, 2*radius+1))
    dark = cv2.erode(min_channel, kernel)
    # Atmosphere light (the average of the brightest %0.1 pixel)
    flat = dark.reshape(-1)
    n_top = max(1, int(0.001 * flat.size))
    idx = np.argpartition(flat, -n_top)[-n_top:]
    A = np.mean(img.reshape(-1,3)[idx], axis=0)
    # Transmission
    t = 1 - omega * (np.min(img / (A + 1e-6), axis=2))
    t = cv2.max(t, t0)

    J = (img - A) / t[...,None] + A
    J = np.clip(J, 0, 1)
    out = (J * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


class VehicleCounter:
    def __init__(self, video_path, show_debug=True,
                 model_name="yolov5mu.pt", device="cuda:0",
                 conf_thres=0.35, iou_thres=0.45, infer_width=640,
                 keep_classes=("car","truck","bus","motorcycle"),
                 match_max_dist=60, track_ttl=10,
                 use_dehaze=False):
        from ultralytics import YOLO
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Video cannot be opened: {video_path}")
        self.W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # detector and tracker
        self.model = YOLO(model_name)
        self.device = device
        self.conf = float(conf_thres)
        self.iou  = float(iou_thres)
        self.imgsz = int(infer_width)
        self.keep = set(keep_classes)
        self.match_max_dist = float(match_max_dist)
        self.track_ttl = int(track_ttl)
        self.use_dehaze = bool(use_dehaze)

        self.next_id = 1
        self.tracks = {}

        # USER ROI VALUES (given manually)
        # (x, y, w, h)
        self.roi_N = (500, 330, 300, 250)
        self.roi_S = (1200, 700, 500, 210)
        self.roi_W = (0, 595, 450, 216)
        self.roi_E = (1329, 460, 698, 100)

        self.show_debug = show_debug

    # Helpers
    @staticmethod
    def in_roi(cx, cy, roi):
        x,y,w,h = roi
        return (x <= cx <= x+w) and (y <= cy <= y+h)


    @staticmethod
    def _dist(a,b):
        return math.hypot(a[0]-b[0], a[1]-b[1])
    

    def _match_tracks(self, detections, iou_dedup_thr=0.7, assign_max_cost=0.85):
        # Pre-dedup
        detections = dedup_detections(detections, iou_thr=iou_dedup_thr)

        used = [False] * len(detections)
        img_diag = math.hypot(self.W, self.H)

        # Update the current tracks(greedy, but based on cost)
        for tid, t in list(self.tracks.items()):
            best_j, best_cost = -1, 1e9
            for j, det in enumerate(detections):
                if used[j]:
                    continue
                cost = _mt_cost(det, t, img_diag)
                if cost < best_cost:
                    best_cost, best_j = cost, j
            if best_j >= 0 and best_cost <= assign_max_cost:
                used[best_j] = True
                cx, cy, xyxy, cname, conf = detections[best_j]
                t['last_pos'] = t['pos']
                t['pos'] = (cx, cy)
                t['ttl'] = self.track_ttl
                t['class'] = cname
                t['conf'] = conf
                t['xyxy'] = xyxy
            else:
                t['ttl'] -= 1
                if t['ttl'] <= 0:
                    self.tracks.pop(tid, None)

        # For new detections to take new ID
        for j, det in enumerate(detections):
            if used[j]:
                continue
            cx, cy, xyxy, cname, conf = det
            self.tracks[self.next_id] = {
                'pos': (cx, cy),
                'last_pos': (cx, cy),
                'ttl': self.track_ttl,
                'class': cname,
                'conf': conf,
                'xyxy': xyxy,
                'ns_side': None,
                'we_side': None
            }
            self.next_id += 1


    def process_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None, None, None, None, None
        # YOLO prediction
        if self.use_dehaze:
            frame = dark_channel_dehaze(frame)

        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf, iou=self.iou,
                                 device=self.device, verbose=False)[0]
        boxes = getattr(res, 'boxes', None)
        names = getattr(res, 'names', {})
        detections = []  # (cx,cy,xyxy,cname,conf)
        cN=cS=cE=cW=0
        if boxes is not None and len(boxes)>0:
            xyxy = boxes.xyxy.cpu().numpy(); cls = boxes.cls.cpu().numpy().astype(int); conf = boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), cid, c in zip(xyxy, cls, conf):
                cname = names.get(int(cid), str(cid))
                if cname not in self.keep: continue
                cx, cy = (x1+x2)/2.0, (y1+y2)/2.0
                detections.append((cx,cy,(float(x1),float(y1),float(x2),float(y2)),cname,float(c)))
                if self.in_roi(cx,cy,self.roi_N):   cN+=1
                elif self.in_roi(cx,cy,self.roi_S): cS+=1
                elif self.in_roi(cx,cy,self.roi_E): cE+=1
                elif self.in_roi(cx,cy,self.roi_W): cW+=1
        self._match_tracks(detections)

        if self.show_debug:
            dbg = frame.copy()
            for roi,color in [(self.roi_N,(0,255,0)),(self.roi_S,(0,255,0)),(self.roi_E,(255,0,0)),(self.roi_W,(255,0,0))]:
                x,y,w,h = roi; cv2.rectangle(dbg,(x,y),(x+w,y+h),color,2)
            for (cx,cy,(x1,y1,x2,y2), cname, c) in detections:
                cv2.rectangle(dbg,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,255),2)
                cv2.putText(dbg,f"{cname} {c:.2f}",(int(x1), max(0,int(y1)-5)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
            cv2.putText(dbg,f"N:{cN} S:{cS} E:{cE} W:{cW}",(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            cv2.imshow("ROI Debug", dbg)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.release(); cv2.destroyAllWindows()
        return frame, cN, cS, cE, cW, detections


# RL Environment (video-based)

def bin_count(x):
    if x <= 0: return 0
    elif x <= 5: return 1
    elif x <= 10: return 2
    elif x <= 20: return 3
    elif x <= 40: return 4
    else: return 5


def bin_duration(t):
    if t <= 5: return 0
    elif t <= 15: return 1
    elif t <= 30: return 2
    elif t <= 60: return 3
    else: return 4


class TrafficEnvFromVideo:
    def __init__(self, video_path, show_debug=True, device="cuda:0"):
        self.id2color = {}
        self.id_where = {}
        self.max_out_len = 7
        self.counter = VehicleCounter(
            video_path,
            show_debug=show_debug,
            model_name="yolov5mu.pt", 
            device=device,
            conf_thres=0.25,
            iou_thres=0.45,
            infer_width=640,
            keep_classes=("car","truck","bus","motorcycle"),
            match_max_dist=60,
            track_ttl=10,
            use_dehaze=False
        )
        self.queues = {
            "N_in": deque(), "S_in": deque(), "E_in": deque(), "W_in": deque(),
            "N_out": deque(), "S_out": deque(), "E_out": deque(), "W_out": deque(),
        }
        self.light_state = {"NS": True, "EW": False}
        self.light_duration = 0
        self.total_reward = 0.0
        self.step_count = 0
        self.phase_budget = 30  # how many cars can be passed in the current phase(simple model)
        self.pending = set() 

        for nm in ("N_out","S_out","E_out","W_out"):
            self.queues[nm] = deque(self.queues[nm], maxlen=self.max_out_len)
    
    # Helpers
    def _color_for_id(self, tid: int) -> str:
        import colorsys
        h = (tid * 137) % 360
        s, v = 0.65, 0.98 # Saturation, value
        r, g, b = colorsys.hsv_to_rgb(h/360.0, s, v) # Normalized by dividing by 360
        return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

    def _ensure_color(self, tid: int) -> str:
        if tid not in self.id2color:
            self.id2color[tid] = self._color_for_id(tid)
        return self.id2color[tid]

    @staticmethod
    def _hex_to_bgr(hexcol: str):
        hexcol = hexcol.lstrip('#')
        r = int(hexcol[0:2], 16); g = int(hexcol[2:4], 16); b = int(hexcol[4:6], 16)
        return (b, g, r)

    def _detect_inbound_ids(self):
        N, S, E, W = [], [], [], []

        # ROI edges
        xN, yN, wN, hN = self.counter.roi_N; n_edge = yN + hN      # Bottom edge for N
        xS, yS, wS, hS = self.counter.roi_S; s_edge = yS            # Top edge for S
        xE, yE, wE, hE = self.counter.roi_E; e_edge = xE            # Left edge for E
        xW, yW, wW, hW = self.counter.roi_W; w_edge = xW + wW       # Right edge for W

        for tid, t in self.counter.tracks.items():
            cx, cy = t['pos']
            lx, ly = t.get('last_pos', (cx, cy))
            dx, dy = cx - lx, cy - ly

            candidate = None
            if self.counter.in_roi(cx, cy, self.counter.roi_N) and dy > 0:
                candidate = ("N", tid, n_edge - cy)
            elif self.counter.in_roi(cx, cy, self.counter.roi_S) and dy < 0:
                candidate = ("S", tid, cy - s_edge)
            elif self.counter.in_roi(cx, cy, self.counter.roi_E) and dx < 0:
                candidate = ("E", tid, cx - e_edge)
            elif self.counter.in_roi(cx, cy, self.counter.roi_W) and dx > 0:
                candidate = ("W", tid, w_edge - cx)

            if candidate:
                lane, tid, dist = candidate
                key = (lane, tid)
                if key in self.pending:
                    # it appears twice → add to inbound
                    if lane=="N": N.append((tid, dist))
                    elif lane=="S": S.append((tid, dist))
                    elif lane=="E": E.append((tid, dist))
                    elif lane=="W": W.append((tid, dist))
                else:
                    # it appears once → add to pending
                    self.pending.add(key)

        # remove from pending: no longer in ROI
        live = {(lane, tid) for lane, tid in self.pending if tid in self.counter.tracks}
        self.pending &= live

        N.sort(key=lambda x:x[1]); S.sort(key=lambda x:x[1])
        E.sort(key=lambda x:x[1]); W.sort(key=lambda x:x[1])
        return [tid for tid,_ in N], [tid for tid,_ in S], [tid for tid,_ in E], [tid for tid,_ in W]

    
    # Helpers
    @staticmethod
    def rand_color():
        palette = ["red","green","blue","yellow","cyan","magenta","orange","purple","brown","pink"]
        return random.choice(palette)

    def _resize_queue_to(self, q: deque, target_len: int):
        cur = len(q)
        if cur < target_len:
            for _ in range(target_len - cur): q.append(self.rand_color())
        elif cur > target_len:
            for _ in range(cur - target_len): q.popleft()

    def update_from_video(self):
        frame, _, _, _, _, _ = self.counter.process_frame()
        if frame is None:
            return None, None, None, None, None

        # Visual overlay (According to the ID color)
        dbg = self.draw_tracks_overlay(frame.copy(), self.counter.tracks)
        cv2.imshow('Tracks', dbg)
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            try:
                self.counter.cap.release()
            except Exception:
                pass
            cv2.destroyAllWindows()

        # New inbound IDs are added
        N_ids, S_ids, E_ids, W_ids = self._detect_inbound_ids()
        for tid in (N_ids + S_ids + E_ids + W_ids):
            self._ensure_color(tid)
        def add_if_new(id_list, qname):
            q = self.queues[qname]
            for tid in id_list:
                if tid not in self.id_where:
                    q.append(tid)
                    self.id_where[tid] = qname
        add_if_new(N_ids, 'N_in')
        add_if_new(S_ids, 'S_in')
        add_if_new(E_ids, 'E_in')
        add_if_new(W_ids, 'W_in')

        # Remove old IDs (No more in VehicleCounter.tracks)
        live = set(self.counter.tracks.keys())
        for qname, q in self.queues.items():
            keep = []
            for tid in q:
                if tid in live:
                    keep.append(tid)
                else:
                    self.id_where.pop(tid, None)
            self.queues[qname].clear()
            self.queues[qname].extend(keep)

        n, s, e, w = len(self.queues['N_in']), len(self.queues['S_in']), len(self.queues['E_in']), len(self.queues['W_in'])
        return frame, n, s, e, w


    def draw_tracks_overlay(self, frame, tracks):
        for tid, t in tracks.items():
            cx, cy = int(t['pos'][0]), int(t['pos'][1])
            col_hex = self.id2color.get(tid, '#00ffaa')
            col_bgr = self._hex_to_bgr(col_hex)
            cv2.circle(frame, (cx, cy), 4, col_bgr, -1)
            cv2.putText(frame, f"ID:{tid}", (cx + 5, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col_bgr, 1)
        return frame


    def move_cars(self, throughput_per_step=2):
        if self.light_state['NS']:
            pairs = [('N_in','S_out'), ('S_in','N_out')]
        else:
            pairs = [('E_in','W_out'), ('W_in','E_out')]
        moved = 0
        for di, do in pairs:
            q_in, q_out = self.queues[di], self.queues[do]
            for _ in range(throughput_per_step):
                if len(q_in) == 0:
                    break
                tid = q_in.popleft()
                # OUT capacity control (7)
                if len(q_out) > self.max_out_len:
                    q_out.popleft()
                q_out.append(tid)
                self.id_where[tid] = do
                moved += 1
        return moved


    def switch_lights(self):
        self.light_state["NS"] = not self.light_state["NS"]
        self.light_state["EW"] = not self.light_state["EW"]
        self.light_duration = 0

    def get_waiting_inbounds(self):
        return len(self.queues["N_in"]) + len(self.queues["S_in"]) + \
               len(self.queues["E_in"]) + len(self.queues["W_in"]) 

    def get_state(self):
        ns_in = len(self.queues["N_in"]) + len(self.queues["S_in"])
        we_in = len(self.queues["E_in"]) + len(self.queues["W_in"])
        return (bin_count(ns_in), bin_count(we_in), int(self.light_state["NS"]), bin_duration(self.light_duration))

    def step(self, action):
        prev_waiting = self.get_waiting_inbounds()
        # 0: move on, 1: change the phase
        if action == 1:
            self.switch_lights()
        self.light_duration += 1
        moved = self.move_cars(throughput_per_step=2)
        self.update_from_video()
        waiting = self.get_waiting_inbounds()
        # Ödül: waiting difference - change penalty
        reward = (prev_waiting - waiting) - (0.05 if action == 1 else 0.0)
        # a little flow reward
        reward += 0.01 * moved
        self.total_reward += reward
        self.step_count += 1
        return self.get_state(), reward


# Q-Learning Agent

class QLearningAgent:
    def __init__(self, n_actions=2, alpha=0.05, gamma=0.95, epsilon=1.0, eps_min=0.05, eps_decay=0.995):
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.Q = defaultdict(lambda: np.zeros(n_actions, dtype=float))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def learn(self, s, a, r, s_next):
        q_sa = self.Q[s][a]
        target = r + self.gamma * np.max(self.Q[s_next])
        self.Q[s][a] = q_sa + self.alpha * (target - q_sa)

    def decay(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# Visualizer

class Visualizer:
    def __init__(self, env: TrafficEnvFromVideo, agent: QLearningAgent, frames=600, interval_ms=1000, show_roi=False):
        self.env = env
        self.agent = agent
        self.frames = frames
        self.interval_ms = interval_ms
        self.show_roi = show_roi
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        plt.axis("off")
        self.state = self.env.update_from_video()  # ilk okuma
        self.state = self.env.get_state()
        self.ani = FuncAnimation(self.fig, self.update, frames=self.frames, interval=self.interval_ms)

    def draw_cross(self):
        self.ax.add_patch(Rectangle((-50,-3), 100,6, color="gray"))
        self.ax.add_patch(Rectangle((-3,-50), 6,100, color="gray"))
        self.ax.add_patch(Rectangle((-3,-3), 6,6, color="black"))

    def draw_lights(self):
        ns_color = "green" if self.env.light_state["NS"] else "red"
        ew_color = "green" if self.env.light_state["EW"] else "red"
        self.ax.add_patch(Circle((0,10), 1, color=ns_color))
        self.ax.add_patch(Circle((0,-10), 1, color=ns_color))
        self.ax.add_patch(Circle((10,0), 1, color=ew_color))
        self.ax.add_patch(Circle((-10,0), 1, color=ew_color))

    def draw_cars(self):
        L, Wd, gap = 2.0, 1.0, 0.6
        max_disp = 7
        q = self.env.queues
        col = self.env.id2color.get
        # inbound
        for i, tid in enumerate(list(q['N_in'])[:max_disp]):
            self.ax.add_patch(Rectangle((-Wd/2-1.7, 3 + i*(L+gap)), Wd, L, color=col(tid, '#aaaaaa')))
        for i, tid in enumerate(list(q['S_in'])[:max_disp]):
            self.ax.add_patch(Rectangle((-Wd/2+1.7, -3 - L - i*(L+gap)), Wd, L, color=col(tid, '#aaaaaa')))
        for i, tid in enumerate(list(q['E_in'])[:max_disp]):
            self.ax.add_patch(Rectangle((3 + i*(L+gap), -Wd/2+1.7), L, Wd, color=col(tid, '#aaaaaa')))
        for i, tid in enumerate(list(q['W_in'])[:max_disp]):
            self.ax.add_patch(Rectangle((-3 - L - i*(L+gap), -Wd/2-1.7), L, Wd, color=col(tid, '#aaaaaa')))
        # outbound (OUT capacity 7)
        for i, tid in enumerate(list(q['N_out'])[:max_disp]):
            self.ax.add_patch(Rectangle((-Wd/2+1.7, 29 - i*(L+gap)), Wd, L, color=col(tid, '#aaaaaa')))
        for i, tid in enumerate(list(q['S_out'])[:max_disp]):
            self.ax.add_patch(Rectangle((-Wd/2-1.7, -30 + i*(L+gap)), Wd, L, color=col(tid, '#aaaaaa')))
        for i, tid in enumerate(list(q['E_out'])[:max_disp]):
            self.ax.add_patch(Rectangle((29 - i*(L+gap), -Wd/2-1.7), L, Wd, color=col(tid, '#aaaaaa')))
        for i, tid in enumerate(list(q['W_out'])[:max_disp]):
            self.ax.add_patch(Rectangle((-30 + i*(L+gap), -Wd/2+1.7), L, Wd, color=col(tid, '#aaaaaa')))


    def hud(self):
        waiting = self.env.get_waiting_inbounds()
        text = (
            f"Step: {self.env.step_count}\n"
            f"Total Reward: {self.env.total_reward:.2f}\n"
            f"Epsilon: {self.agent.epsilon:.3f}\n"
            f"Light NS: {'G' if self.env.light_state['NS'] else 'R'}  Dur: {self.env.light_duration}\n"
            f"Waiting(in): {waiting}"
        )
        self.ax.text(-29, 27, text, fontsize=8, va='top', ha='left',
                     bbox=dict(facecolor='white', alpha=0.6, boxstyle='round,pad=0.3'))
    
    def draw_road_markings(self):
        for y in range(-50, 50, 5):
            self.ax.plot([0, 0], [y, y+2], color="white", linewidth=2, solid_capstyle="butt", zorder=1)
        for x in range(-50, 50, 5):
            self.ax.plot([x, x+2], [0, 0], color="white", linewidth=2, solid_capstyle="butt", zorder=1)

    def update(self, frame_idx):
        s = self.env.get_state()
        a = self.agent.choose_action(s)
        s2, r = self.env.step(a)
        self.agent.learn(s, a, r, s2)
        self.agent.decay()
        self.ax.clear(); self.ax.set_xlim(-30,30); self.ax.set_ylim(-30,30); self.ax.axis("off")
        self.draw_cross(); self.draw_road_markings(); self.draw_cars(); self.draw_lights(); self.hud()


# Execution of the program

if __name__ == "__main__":
    VIDEO = r"C:\\Users\\Administrator\\Downloads\\camera_junction2.mp4"  # Video path on the PC
    DEVICE = "cuda:0" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

    env = TrafficEnvFromVideo(VIDEO, show_debug=True, device=DEVICE)
    agent = QLearningAgent(n_actions=2, alpha=0.05, gamma=0.95, epsilon=1.0, eps_min=0.05, eps_decay=0.995)

    viz = Visualizer(env, agent, frames=800, interval_ms=800, show_roi=False)
    plt.show()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass