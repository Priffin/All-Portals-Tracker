import os
import bisect
import json
import time
import threading
import numpy as np
import pyperclip
import requests
import contextlib
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from lincolnsolver import Pathfinding


class StrongholdTracker:
    STRONGHOLD_RINGS = [
        (1280, 2816),
        (4352, 5888),
        (7424, 8960),
        (10496, 12032),
        (13568, 15104),
        (16640, 18176),
        (19712, 21248),
        (22784, 24320),
    ]
    MAX_STRONGHOLDS = 129

    def __init__(self):
        # Configuration
        self.app = Flask(__name__)
        self.app.config["SECRET_KEY"] = "secret!"
        self.socketio = SocketIO(self.app)

        # State variables
        self.ninbot_coords = None
        self.route = None
        self.target_coords = None
        self.skips = 0
        self.first_eight_strongholds = []
        self.stronghold_count = 0

        # Initial UI values
        self.values = {
            "coords": "Waiting for pathfinding",
            "distance": "",
            "angle": "",
            "stronghold_count": "0/129",
            "instructions": "",
        } | {f"ring{i}": "" for i in range(1, 9)}

        # Flask & SocketIO routing
        self.app.add_url_rule("/", None, self.flask_index)
        self.socketio.on_event("connect", self.on_client_connect)
        self.socketio.on_event("disconnect", self.on_client_disconnect)

    def flask_index(self):
        return render_template("index.html", values=self.values)

    def on_client_connect(self):
        print("Client connected")
        self.socketio.emit("update_values", self.values)
        self.socketio.emit("toggle_tablegraph", "table")
        self.socketio.emit("clear_points")
        for i in range(1, 9):
            self.update_number(f"ring{i}", "")

    def on_client_disconnect(self):
        print("Client disconnected")

    def save_backup(self, filepath="all_portals_backup.json"):
        with open(filepath, "w+", encoding="utf-8") as backup_file:
            json.dump([t[1] for t in self.first_eight_strongholds], backup_file, indent=2)

    def load_backup(self, filepath="all_portals_backup.json"):
        with open(filepath, "r", encoding="utf-8") as backup_file:
            data = json.load(backup_file)
        for i, coords in enumerate(data[:8]):
            coords = tuple(coords)
            self.update_number(f"ring{i+1}", str(coords))
            self.add_stronghold(coords, save=False)

    def add_stronghold(self, coords, save=True):
        if coords:
            ring = self.get_stronghold_ring(coords)
            bisect.insort(self.first_eight_strongholds, (ring, coords))
            self.update_number(f"ring{ring}", str(coords))

        if len(self.first_eight_strongholds) == 8:
            if save:
                self.save_backup()

            for ring, (x, y) in self.first_eight_strongholds:
                self.socketio.emit(
                    "generate_point", (x, -y, f"p{self.stronghold_count}", "#8a0b11")
                )

            scaled_coords = [
                (t[1][0] * 8, t[1][1] * 8) for t in self.first_eight_strongholds
            ]
            solver = Pathfinding()
            self.route = solver.make_stronghold_list(scaled_coords)

            for r in self.route:
                print(tuple(map(lambda x: int(x // 8), r[0])), r[-1])

            self.socketio.emit("toggle_tablegraph", "graph")
            self.next_stronghold()

    def next_stronghold(self):
        sh = self.route[self.stronghold_count - 9]
        self.target_coords = tuple(map(lambda x: int(x // 8), sh[0]))

        self.socketio.emit(
            "generate_point",
            (
                self.target_coords[0],
                -self.target_coords[1],
                f"p{self.stronghold_count}",
            ),
        )
        self.update_number("coords", str(self.target_coords))

        instructions = {
            2: "Do not set your spawnpoint at this stronghold.",
            1: "Leave your bed at this stronghold.",
        }
        self.update_number("instructions", instructions.get(sh[-1], ""))

    def update_number(self, number_id, new_value):
        if number_id in self.values:
            self.values[number_id] = new_value
            self.socketio.emit("update_values", self.values)
            return True
        return False

    def get_stronghold_ring(self, coords):
        dist = self.distance_between_points((0, 0), (coords[0] * 8, coords[1] * 8))
        return next(
            (
                ring + 1
                for ring, bounds in enumerate(self.STRONGHOLD_RINGS)
                if bounds[0] < dist < bounds[1]
            ),
            0,
        )

    @staticmethod
    def get_360_angle(a):
        while not (-180 <= a <= 180):
            a = a + 360 if a < -180 else a - 360
        return a

    @staticmethod
    def calc_angle(x1, y1, x2, y2):
        return np.degrees(np.arctan2(y2 - y1, x2 - x1) if (x1, y1) != (x2, y2) else 0)

    @staticmethod
    def smallest_angle_difference(angle1, angle2):
        diff = (angle1 - angle2) % 360
        if diff > 180:
            diff -= 360
        return round(diff, 1)

    @staticmethod
    def distance_between_points(p1, p2):
        return np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def get_record_path(self):
        home_dir = os.path.expanduser("~")
        latest_world_dir = os.path.join(home_dir, "speedrunigt", "latest_world.json")
        try:
            with open(latest_world_dir, "r") as f:
                data = json.load(f)
                return os.path.join(
                    data.get("world_path"), "speedrunigt", "record.json"
                )
        except Exception as e:
            print(f"Error getting record path: {e}")
            return None

    def count_strongholds(self, filepath):
        old_count = self.stronghold_count
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                try:
                    self.stronghold_count = (
                        max(
                            int(t["name"][10:])
                            for t in data.get("timelines")
                            if t["name"][:10] == "portal_no_"
                        )
                        + self.skips
                    )
                except ValueError:
                    self.stronghold_count = 0

                self.update_number(
                    "stronghold_count",
                    f"{self.stronghold_count}/{self.MAX_STRONGHOLDS}",
                )

                if old_count != self.stronghold_count:
                    if 1 <= self.stronghold_count <= 8:
                        self.add_stronghold(self.ninbot_coords)
                    elif self.stronghold_count >= 9:
                        self.next_stronghold()
                    elif self.stronghold_count == 0:
                        self.update_number(
                            "stronghold_count",
                            f"{self.stronghold_count}/{self.MAX_STRONGHOLDS}",
                        )
                        self.socketio.emit("toggle_tablegraph", "table")
                        self.socketio.emit("clear_points")
                        for i in range(1, 9):
                            self.update_number(f"ring{i}", "")
        except Exception as e:
            print(f"Error counting strongholds: {e}")

    def monitor_file(self):
        filepath = self.get_record_path()
        last_modified = 0
        if filepath and os.path.isfile(filepath):
            last_modified = os.path.getmtime(filepath)
            self.count_strongholds(filepath)

        while True:
            filepath = self.get_record_path()
            if not filepath or not os.path.isfile(filepath):
                time.sleep(0.5)
                continue

            try:
                current_modified = os.path.getmtime(filepath)
                if current_modified != last_modified:
                    self.count_strongholds(filepath)
                    last_modified = current_modified
            except FileNotFoundError:
                print(f"File not found: {filepath}")
                return
            time.sleep(0.5)

    def monitor_clipboard(self):
        previous = pyperclip.paste()
        while True:
            time.sleep(0.1)
            try:
                contents = pyperclip.paste()
                if contents == previous:
                    continue
                previous = contents

                if contents == "+skip":
                    self.stronghold_count += 1
                    self.skips += 1
                    self.update_number(
                        "stronghold_count",
                        f"{self.stronghold_count}/{self.MAX_STRONGHOLDS}",
                    )
                    self.next_stronghold()

                if contents == "+load_backup":
                    with contextlib.suppress(FileNotFoundError):
                        self.load_backup()

                if not contents.startswith("/execute in"):
                    continue

                components = contents.split()
                dimension = components[2][10:]
                coords = {
                    "x": (
                        float(components[6]) / 8
                        if dimension == "overworld"
                        else float(components[6])
                    ),
                    "y": float(components[7]),
                    "z": (
                        float(components[8]) / 8
                        if dimension == "overworld"
                        else float(components[8])
                    ),
                    "a": float(components[9]),
                }

                if self.target_coords:
                    sh_x, sh_z = self.target_coords
                    s_a = self.calc_angle(coords["x"], coords["z"], sh_x, sh_z)
                    stronghold_angle = self.get_360_angle(-90 + s_a)
                    angle_delta = self.smallest_angle_difference(
                        self.get_360_angle(coords["a"]), stronghold_angle
                    )

                    angle_text = f'{"<--" if angle_delta > 0 else ""}{abs(angle_delta)}{"-->" if angle_delta <= 0 else ""}'
                    distance = self.distance_between_points(
                        (coords["x"], coords["z"]), (sh_x, sh_z)
                    )

                    self.update_number(
                        "angle", f"Angle: {round(stronghold_angle, 1)} ({angle_text})"
                    )
                    self.update_number("distance", f"Distance: {round(distance)}")

            except pyperclip.PyperclipException as e:
                print(f"Error accessing clipboard: {e}")
                break
            except KeyboardInterrupt:
                print("\nExiting clipboard monitor.")
                break

    def monitor_ninbot(self):
        url = "http://localhost:52533/api/v1/stronghold/events"
        while True:
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            line = json.loads(line.decode("utf-8")[6:])
                            stronghold = line["predictions"][0]
                            self.ninbot_coords = (
                                stronghold["chunkX"] * 2,
                                stronghold["chunkZ"] * 2,
                            )
                        except Exception as e:
                            pass
            except requests.exceptions.RequestException as e:
                print(f"Error connecting to event stream: {e}")
                time.sleep(10)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

    def start_monitoring(self):
        threads = [
            threading.Thread(target=self.monitor_ninbot, daemon=True),
            threading.Thread(target=self.monitor_file, daemon=True),
            threading.Thread(target=self.monitor_clipboard, daemon=True),
        ]
        for thread in threads:
            thread.start()

    def run(self):
        self.start_monitoring()
        self.socketio.run(self.app, debug=True, port=5123)


def main():
    tracker = StrongholdTracker()
    tracker.run()


if __name__ == "__main__":
    main()
