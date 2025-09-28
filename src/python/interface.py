import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import subprocess
import sys
import os
import threading
import signal

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PAUSE_SCRIPT = os.path.join(SCRIPT_DIR, "pause.py")
CALIBRATE_SCRIPT = os.path.join(SCRIPT_DIR, "calibrate.py")


class AppRunner:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Pauser Control")
        self.root.geometry("700x500")
        self.time_geek = 0.0        # total paused time (or whatever you want to track)
        self.pause_start = None     # when we entered PAUSED


        self.calibrated_yaw = None
        self.process: subprocess.Popen | None = None
        self.paused = False

        # --- Main layout containers ---
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left: vertical controls
        sidebar = tk.Frame(main_frame, padx=8, pady=8)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)

        # Right: status + logs
        content = tk.Frame(main_frame)
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Buttons (stacked vertically) ---
        self.calibrate_button = tk.Button(
            sidebar, text="1. Calibrate Center Position",
            command=self.run_calibration, font=("Arial", 12)
        )
        self.calibrate_button.pack(fill=tk.X, pady=4)

        self.start_button = tk.Button(
            sidebar, text="2. Run Smart Pauser",
            command=self.run_pauser, state=tk.DISABLED, font=("Arial", 12)
        )
        self.start_button.pack(fill=tk.X, pady=4)

        self.pause_button = tk.Button(
            sidebar, text="3. Pause",
            command=self.pause_script, state=tk.DISABLED, font=("Arial", 12)
        )
        self.pause_button.pack(fill=tk.X, pady=4)

        self.resume_button = tk.Button(
            sidebar, text="4. Resume",
            command=self.resume_script, state=tk.DISABLED, font=("Arial", 12)
        )
        self.resume_button.pack(fill=tk.X, pady=4)

        self.kill_button = tk.Button(
            sidebar, text="5. End",
            command=self.kill_script, state=tk.DISABLED, font=("Arial", 12)
        )
        self.kill_button.pack(fill=tk.X, pady=4)

        self.quit_button = tk.Button(
            sidebar, text="Quit Application",
            command=self.stop_program, state=tk.NORMAL, font=("Arial", 12)
        )
        self.quit_button.pack(fill=tk.X, pady=12)

        # --- Status + output (right side) ---
        self.status_label = tk.Label(
            content, text="Status: Please calibrate first.",
            fg="red", font=("Arial", 10)
        )
        self.status_label.pack(pady=(10, 4), anchor="w")

        self.text_output = ScrolledText(
            content, height=25, width=90, state="disabled",
            bg="black", fg="lightgray"
        )
        self.text_output.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Window close handler
        self.root.protocol("WM_DELETE_WINDOW", self.stop_program)

    # -------- helpers ----------
    def log(self, message: str):
        self.text_output.config(state="normal")
        self.text_output.insert(tk.END, message)
        self.text_output.see(tk.END)
        self.text_output.config(state="disabled")

    def _reset_buttons_to_idle(self):
        self.start_button.config(state=tk.NORMAL)
        self.calibrate_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.resume_button.config(state=tk.DISABLED)
        self.kill_button.config(state=tk.DISABLED)

    def _set_running_buttons(self):
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.resume_button.config(state=tk.DISABLED)
        self.kill_button.config(state=tk.NORMAL)

    # -------- calibration ----------
    def run_calibration(self):
        self.log("[INFO] Starting calibration...\n")
        self.calibrate_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Calibration in progress...", fg="orange")

        try:
            self.process = subprocess.Popen(
                [sys.executable, CALIBRATE_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge to avoid deadlock
                text=True,
                bufsize=1
            )
            threading.Thread(target=self._wait_for_calibration, daemon=True).start()
        except Exception as e:
            self.log(f"[ERROR] Failed to launch calibration: {e}\n")
            self.calibrate_button.config(state=tk.NORMAL)

    def _wait_for_calibration(self):
        assert self.process is not None
        stdout, _ = self.process.communicate()
        returncode = self.process.returncode
        self.root.after(0, self._handle_calibration_result, stdout, returncode)

    def _handle_calibration_result(self, stdout: str, returncode: int):
        if returncode == 0:
            try:
                yaw_value_str = stdout.strip().splitlines()[-1]
                self.calibrated_yaw = float(yaw_value_str)
                self.log(f"[SUCCESS] Calibration complete. Center Yaw: {self.calibrated_yaw:.2f}\n")
                self.status_label.config(
                    text=f"Status: Calibrated (center={self.calibrated_yaw:.2f}). Ready to run.",
                    fg="green"
                )
                self.start_button.config(state=tk.NORMAL)
            except (ValueError, IndexError):
                self.log("[ERROR] Could not parse calibration value.\n")
                self.status_label.config(text="Status: Calibration failed. Try again.", fg="red")
        else:
            self.log("[ERROR] Calibration script failed or was cancelled.\n")
            self.status_label.config(text="Status: Calibration failed. Try again.", fg="red")

        self.calibrate_button.config(state=tk.NORMAL)
        self.process = None

    # -------- runner ----------
    def run_pauser(self):
        if self.calibrated_yaw is None:
            self.log("[ERROR] Please calibrate first.\n")
            return

        self.log(f"[INFO] Starting Smart Pauser (center {self.calibrated_yaw:.2f})...\n")
        self._set_running_buttons()
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)
        self.paused = False

        try:
            self.process = subprocess.Popen(
                [sys.executable, PAUSE_SCRIPT, str(self.calibrated_yaw)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # merge to avoid blocking on stderr
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            threading.Thread(target=self.stream_output, args=(self.process.stdout,), daemon=True).start()
        except Exception as e:
            self.log(f"[ERROR] Failed to launch Smart Pauser: {e}\n")
            self._reset_buttons_to_idle()
            self.process = None

    def stream_output(self, pipe):
        for line in pipe:
            self.root.after(0, self.log, line)
        pipe.close()
        self.root.after(100, self._on_process_exit)

    def _on_process_exit(self):
        if self.process:
            rc = self.process.poll()
            if rc is None:
                # still running; check again
                self.root.after(100, self._on_process_exit)
                return
            self.log(f"[INFO] Smart Pauser exited with code {rc}.\n")
        self._reset_buttons_to_idle()
        self.process = None
        self.paused = False

    # -------- controls ----------
    def pause_script(self):
        if not self.process or self.process.poll() is not None:
            return
        try:
            self.process.send_signal(signal.SIGSTOP)
            self.paused = True
            self.log("[INFO] Paused.\n")
            self.pause_button.config(state=tk.DISABLED)
            self.resume_button.config(state=tk.NORMAL)
        except Exception as e:
            self.log(f"[ERROR] Pause failed: {e}\n")

    def resume_script(self):
        if not self.process or self.process.poll() is not None:
            return
        try:
            self.process.send_signal(signal.SIGCONT)
            self.paused = False
            self.log("[INFO] Resumed.\n")
            self.resume_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
        except Exception as e:
            self.log(f"[ERROR] Resume failed: {e}\n")

    def kill_script(self):
        if not self.process or self.process.poll() is not None:
            return
        try:
            self.process.terminate()
            self.log("[INFO] Terminating...\n")
        except Exception as e:
            self.log(f"[ERROR] Terminate failed: {e}\n")

    def stop_program(self):
        try:
            self.kill_script()
        finally:
            self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = AppRunner(root)
    root.mainloop()
