import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import subprocess
import sys
import os
<<<<<<< HEAD
import threading

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PAUSE_SCRIPT = os.path.join(SCRIPT_DIR, "pause.py")
CALIBRATE_SCRIPT = os.path.join(SCRIPT_DIR, "calibrate.py")

class AppRunner:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Pauser Control")
        self.root.geometry("700x500")

        self.calibrated_yaw = None
        self.process = None

        # --- Widgets ---
        control_frame = tk.Frame(root)
        control_frame.pack(pady=10)

        self.calibrate_button = tk.Button(control_frame, text="1. Calibrate Center Position", command=self.run_calibration, font=("Arial", 12))
        self.calibrate_button.pack(side=tk.LEFT, padx=10)

        self.start_button = tk.Button(control_frame, text="2. Run Smart Pauser", command=self.run_pauser, state=tk.DISABLED, font=("Arial", 12))
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.status_label = tk.Label(root, text="Status: Please calibrate first.", fg="red", font=("Arial", 10))
        self.status_label.pack(pady=5)

        self.text_output = ScrolledText(root, height=25, width=90, state="disabled", bg="black", fg="lightgray")
        self.text_output.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def log(self, message):
        """Inserts a message into the text output widget on the main thread."""
        self.text_output.config(state="normal")
        self.text_output.insert(tk.END, message)
        self.text_output.see(tk.END)
        self.text_output.config(state="disabled")

    def run_calibration(self):
        """Runs the calibration script and captures its output."""
        self.log("[INFO] Starting calibration...\n")
        self.calibrate_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Calibration in progress...", fg="orange")

        try:
            self.process = subprocess.Popen(
                [sys.executable, CALIBRATE_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            # This worker thread will wait for the process to finish
            threading.Thread(target=self._wait_for_calibration, daemon=True).start()

        except Exception as e:
            self.log(f"[ERROR] Failed to launch calibration: {e}\n")
            self.calibrate_button.config(state=tk.NORMAL)

    def _wait_for_calibration(self):
        """
        [WORKER THREAD] Waits for the calibration process to complete and then
        schedules the UI update on the main thread.
        """
        stdout, stderr = self.process.communicate()
        returncode = self.process.returncode
        
        # FIX: Schedule the UI update to run on the main GUI thread
        self.root.after(0, self._handle_calibration_result, stdout, stderr, returncode)

    def _handle_calibration_result(self, stdout, stderr, returncode):
        """
        [MAIN THREAD] Safely updates the UI with the calibration result.
        """
        if returncode == 0:
            try:
                # The yaw value is the last line of stdout
                yaw_value_str = stdout.strip().split('\n')[-1]
                self.calibrated_yaw = float(yaw_value_str)
                self.log(f"[SUCCESS] Calibration complete. Center Yaw set to: {self.calibrated_yaw:.2f}\n")
                self.status_label.config(text=f"Status: Calibrated! Center Yaw = {self.calibrated_yaw:.2f}. Ready to run.", fg="green")
                self.start_button.config(state=tk.NORMAL)
            except (ValueError, IndexError):
                self.log(f"[ERROR] Could not read calibration value from script output.\n")
                self.status_label.config(text="Status: Calibration failed. Please try again.", fg="red")
        else:
            self.log(f"[ERROR] Calibration script failed or was cancelled.\n")
            self.log(stderr) # Log any errors from the script
            self.status_label.config(text="Status: Calibration failed. Please try again.", fg="red")
        
        self.calibrate_button.config(state=tk.NORMAL)


    def run_pauser(self):
        """Runs the main smart pauser script in a separate process."""
        self.log(f"[INFO] Starting Smart Pauser with center yaw {self.calibrated_yaw:.2f}...\n")
        self.start_button.config(state=tk.DISABLED)
        self.calibrate_button.config(state=tk.DISABLED)

        try:
            self.process = subprocess.Popen(
                [sys.executable, PAUSE_SCRIPT, str(self.calibrated_yaw)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1, # Line-buffered
                universal_newlines=True
            )
            # Start threads to stream output in real-time
            threading.Thread(target=self.stream_output, args=(self.process.stdout,), daemon=True).start()
            threading.Thread(target=self.stream_output, args=(self.process.stderr,), daemon=True).start()
        except Exception as e:
            self.log(f"[ERROR] Failed to launch Smart Pauser: {e}\n")
            self.start_button.config(state=tk.NORMAL)
            self.calibrate_button.config(state=tk.NORMAL)


    def stream_output(self, pipe):
        """
        [WORKER THREAD] Reads output from a process pipe line-by-line and
        schedules the logging to happen on the main thread.
        """
        for line in pipe:
            # FIX: Schedule the log message to be displayed by the main thread
            self.root.after(0, self.log, line)
        pipe.close()
        # Check if the process is finished and re-enable buttons
        self.root.after(100, self._check_process_and_enable_buttons)


    def _check_process_and_enable_buttons(self):
        """
        [MAIN THREAD] Checks if the process has finished and re-enables buttons.
        """
        if self.process and self.process.poll() is not None:
             self.log("\n[INFO] Smart Pauser script has exited.\n")
             self.start_button.config(state=tk.NORMAL)
             self.calibrate_button.config(state=tk.NORMAL)

    def on_closing(self):
        """Handle window closing."""
        if self.process and self.process.poll() is None:
            self.process.terminate() # Ensure child process is killed
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = AppRunner(root)
    root.mainloop()

=======

# Path to the script you want to run
SCRIPT = os.path.join(os.path.dirname(__file__), "pause.py")
# SCRIPT = os.path.join(os.path.dirname(__file__), "test.py")

def run_script():
    # Disable button during run
    start_button.config(state=tk.DISABLED)
    text_output.config(state="normal")
    text_output.delete("1.0", tk.END)

    try:
        result = subprocess.run(
            [sys.executable, SCRIPT],
            capture_output=True,
            text=True
        )
        text_output.insert(tk.END, result.stdout)
        text_output.insert(tk.END, f"\n[Exited with code {result.returncode}]\n")
    except Exception as e:
        text_output.insert(tk.END, f"Error: {e}\n")
    finally:
        text_output.config(state="disabled")
        start_button.config(state=tk.NORMAL)

# GUI setup
root = tk.Tk()
root.title("Blocking Subprocess Example")
root.geometry("600x400")

start_button = tk.Button(root, text="Run Script", command=run_script)
start_button.pack(pady=10)

text_output = ScrolledText(root, height=20, width=80, state="disabled")
text_output.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

root.mainloop()
>>>>>>> 82334f5 (changed interface file)
