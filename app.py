import os
import bisect
import json
import time
import threading
import logging
import contextlib
import numpy as np
import pyperclip
import requests
from flask import Flask, render_template
from flask_socketio import SocketIO
from route_solver import ORSolver, PuLPRingStarSolver


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

        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

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
        self.logger.info("SocketIO client connected")
        self.socketio.emit("update_values", self.values)
        self.socketio.emit("toggle_tablegraph", "table")
        self.socketio.emit("clear_graph")
        for i in range(1, 9):
            self.update_number(f"ring{i}", "")

    def on_client_disconnect(self):
        self.logger.info("SocketIO client disconnected")

    def save_backup(self, filepath="all_portals_backup.json"):
        with open(filepath, "w+", encoding="utf-8") as backup_file:
            json.dump(
                [t[1] for t in self.first_eight_strongholds], backup_file, indent=2
            )

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
                angle = np.arctan2(y, x)
                distance = (
                    self.STRONGHOLD_RINGS[ring - 1][0]
                    + self.STRONGHOLD_RINGS[ring - 1][1]
                ) // 2
                x = distance * np.cos(angle) // 8
                y = distance * np.sin(angle) // 8
                self.socketio.emit(
                    "generate_point", (x, -y, f"p{self.stronghold_count}", "#8a0b11")
                )

            scaled_coords = [
                (t[1][0] * 8, t[1][1] * 8) for t in self.first_eight_strongholds
            ]
            self.route = PuLPRingStarSolver.solve(scaled_coords)

            self.socketio.emit("toggle_tablegraph", "graph")
            self.next_stronghold()

    def next_stronghold(self):
        if self.route is None:
            return
        if self.stronghold_count < 8:
            self.logger.info("Not enough strongholds to route through")
            return
        if self.stronghold_count >= self.MAX_STRONGHOLDS:
            self.logger.info("Path complete")
            self.update_number("coords", "All portals complete!")
            return
        sh = self.route[self.stronghold_count - 8]
        self.target_coords = (sh[0][0] // 8, sh[0][1] // 8)

        self.socketio.emit(
            "generate_point",
            (
                self.target_coords[0],
                -self.target_coords[1],
                f"p{self.stronghold_count}",
            ),
        )
        self.socketio.emit(
            "generate_line",
            (
                sh[3][0] // 8,
                -(sh[3][1] // 8),
                sh[2][0] // 8,
                -(sh[2][1] // 8),
                sh[5],
                15 if sh[-1] == 2 else 40,
            ),
        )
        self.update_number("coords", str(self.target_coords))

        instructions = {
            3: "Take your bed but do not set your spawnpoint at this stronghold.",
            2: "Do not take your bed.",
            1: "Take your bed and leave it at this stronghold.",
            0: "Take your bed and set your spawnpoint at this stronghold.",
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
            with open(latest_world_dir, "r", encoding="utf-8") as f:
                data = json.load(f)
                return os.path.join(
                    data.get("world_path"), "speedrunigt", "record.json"
                )
        except (FileNotFoundError, KeyError):
            logging.getLogger("record_monitor").error(
                "Error getting record path", exc_info=True
            )
            return None

    def count_strongholds(self, filepath):
        old_count = self.stronghold_count
        try:
            with open(filepath, "r", encoding="utf-8") as f:
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
                        self.socketio.emit("clear_graph")
                        for i in range(1, 9):
                            self.update_number(f"ring{i}", "")
        except (FileNotFoundError, KeyError):
            logging.getLogger("record_monitor").error(
                "Error counting strongholds", exc_info=True
            )

    def monitor_file(self):
        logger = logging.getLogger("record_monitor")
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
                logger.error("Record file not found", exc_info=True)
                return
            time.sleep(0.5)

    def monitor_clipboard(self):
        logger = logging.getLogger("clipboard_monitor")
        previous = pyperclip.paste()
        while True:
            time.sleep(0.1)
            try:
                contents = pyperclip.paste()
                if contents == previous:
                    continue
                previous = contents

                if contents == "+skip":
                    # clear clipboard to make repeated skips easier
                    pyperclip.copy("")
                    self.logger.info("Skipping stronghold")
                    self.stronghold_count += 1
                    self.skips += 1
                    self.update_number(
                        "stronghold_count",
                        f"{self.stronghold_count}/{self.MAX_STRONGHOLDS}",
                    )
                    self.next_stronghold()

                if contents == "+load_backup":
                    self.logger.info("Loading backup")
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

            except pyperclip.PyperclipException:
                logger.error("Error accessing clipboard", exc_info=True)
                break
            except KeyboardInterrupt:
                logger.info("Exiting clipboard monitor.", exc_info=True)
                break

    def monitor_ninbot(self):
        logger = logging.getLogger("ninbot_monitor")
        url = "http://localhost:52533/api/v1/stronghold/events"
        while True:
            try:
                response = requests.get(url, stream=True, timeout=10)
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            line = json.loads(line.decode("utf-8")[6:])
                            if line["predictions"]:
                                stronghold = line["predictions"][0]
                                self.ninbot_coords = (
                                    stronghold["chunkX"] * 2,
                                    stronghold["chunkZ"] * 2,
                                )
                        except (KeyError, IndexError, json.JSONDecodeError):
                            logger.error(
                                "Error parsing NinjabrainBot API response",
                                exc_info=True,
                            )
            except requests.exceptions.RequestException:
                time.sleep(10)

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
        self.socketio.run(self.app, debug=True, port=5123, use_reloader=False)


def main():
    tracker = StrongholdTracker()
    tracker.run()


if __name__ == "__main__":
    main()
