"""
arm_shm_bridge.py — Shared-memory sidecar for xr_teleoperate → MuJoCo arm control.

Layout (all float64):
  [0]      : timestamp
  [1]      : sequence number
  [2:16]   : sol_q (14 floats — 7 left arm + 7 right arm)
  [16:30]  : sol_tauff (14 floats)
  [30]     : valid flag (1.0 = data written)
  [31]     : reset flag (1.0 = reset requested, cleared after read)

Total: 32 doubles = 256 bytes
"""

import mmap
import os
import struct
import time
import numpy as np

SHM_PATH = "/dev/shm/xr_teleop_arm_bridge"
NUM_ARM_JOINTS = 14
HEADER_FIELDS = 2
FOOTER_FIELDS = 2  # valid + reset
TOTAL_DOUBLES = HEADER_FIELDS + NUM_ARM_JOINTS * 2 + FOOTER_FIELDS  # 32
SHM_SIZE = TOTAL_DOUBLES * 8  # 256 bytes
STRUCT_FMT = f"<{TOTAL_DOUBLES}d"


class ArmShmWriter:
    def __init__(self):
        fd = os.open(SHM_PATH, os.O_CREAT | os.O_RDWR, 0o666)
        os.ftruncate(fd, SHM_SIZE)
        self.mm = mmap.mmap(fd, SHM_SIZE)
        os.close(fd)
        self.seq = 0
        self.mm.seek(0)
        self.mm.write(b'\x00' * SHM_SIZE)
        print(f"[ArmShmWriter] Shared memory created at {SHM_PATH} ({SHM_SIZE} bytes)")

    def write(self, sol_q, sol_tauff):
        self.seq += 1
        q = list(sol_q[:NUM_ARM_JOINTS])
        tau = list(sol_tauff[:NUM_ARM_JOINTS])
        data = struct.pack(
            STRUCT_FMT,
            time.time(),
            float(self.seq),
            *q,
            *tau,
            1.0,  # valid
            0.0   # reset (not requesting)
        )
        self.mm.seek(0)
        self.mm.write(data)

    def request_reset(self):
        """Signal MuJoCo to reset the environment."""
        self.seq += 1
        data = struct.pack(
            STRUCT_FMT,
            time.time(),
            float(self.seq),
            *([0.0] * NUM_ARM_JOINTS),  # zero joint targets
            *([0.0] * NUM_ARM_JOINTS),  # zero torques
            1.0,  # valid
            1.0   # reset flag
        )
        self.mm.seek(0)
        self.mm.write(data)

    def close(self):
        self.mm.close()
        try:
            os.unlink(SHM_PATH)
        except OSError:
            pass


class ArmShmReader:
    def __init__(self, timeout_sec=0.5):
        self.timeout_sec = timeout_sec
        self.mm = None
        self.last_seq = -1
        self._open()

    def _open(self):
        try:
            fd = os.open(SHM_PATH, os.O_RDONLY)
            self.mm = mmap.mmap(fd, SHM_SIZE, access=mmap.ACCESS_READ)
            os.close(fd)
            print(f"[ArmShmReader] Connected to shared memory at {SHM_PATH}")
        except FileNotFoundError:
            self.mm = None

    def read(self):
        """
        Returns (sol_q, sol_tauff, is_new, is_valid, reset_requested)
        """
        if self.mm is None:
            self._open()
            if self.mm is None:
                return np.zeros(NUM_ARM_JOINTS), np.zeros(NUM_ARM_JOINTS), False, False, False

        self.mm.seek(0)
        raw = self.mm.read(SHM_SIZE)
        vals = struct.unpack(STRUCT_FMT, raw)

        timestamp = vals[0]
        seq = int(vals[1])
        sol_q = np.array(vals[2:2 + NUM_ARM_JOINTS])
        sol_tauff = np.array(vals[2 + NUM_ARM_JOINTS:2 + 2 * NUM_ARM_JOINTS])
        valid = vals[-2]
        reset = vals[-1]

        is_valid = (valid == 1.0) and (time.time() - timestamp < self.timeout_sec)
        is_new = (seq != self.last_seq)
        self.last_seq = seq

        return sol_q, sol_tauff, is_new, is_valid, (reset == 1.0)

    def close(self):
        if self.mm is not None:
            self.mm.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "write":
        writer = ArmShmWriter()
        print("Writing test data every 33ms. Press Ctrl+C to stop.")
        i = 0
        try:
            while True:
                t = i * 0.033
                q = [0.1 * np.sin(t + j) for j in range(NUM_ARM_JOINTS)]
                tau = [0.0] * NUM_ARM_JOINTS
                writer.write(q, tau)
                i += 1
                time.sleep(0.033)
        except KeyboardInterrupt:
            writer.close()
    elif len(sys.argv) > 1 and sys.argv[1] == "reset":
        writer = ArmShmWriter()
        writer.request_reset()
        print("Reset requested.")
        writer.close()
    else:
        reader = ArmShmReader()
        print("Reading shared memory. Press Ctrl+C to stop.")
        try:
            while True:
                sol_q, sol_tauff, is_new, is_valid, reset = reader.read()
                if reset:
                    print("[RESET REQUESTED]")
                elif is_valid:
                    status = "NEW" if is_new else "old"
                    print(f"[{status}] q={sol_q[:4].round(3)}...")
                else:
                    print("[waiting]")
                time.sleep(0.033)
        except KeyboardInterrupt:
            reader.close()
