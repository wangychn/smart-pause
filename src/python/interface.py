import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import subprocess
import sys
import os
import threading
import subprocess, sys, os, signal, threading, queue

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PAUSE_SCRIPT = os.path.join(SCRIPT_DIR, "pause.py")
CALIBRATE_SCRIPT = os.path.join(SCRIPT_DIR, "calibrate.py")


proc = None
out_q: "queue.Queue[str|None]" = queue.Queue()

def reader_thread(p: subprocess.Popen, q: "queue.Queue"):
    # Stream stdout without blocking the Tk thread
    try:
        for line in p.stdout:
            q.put(line)
    finally:
        q.put(None)  # sentinel: process ended


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

        self.pause_button = tk.Button(control_frame, text="3. Pause Smart Pauser", command=self.pause_script, state=tk.DISABLED, font=("Arial", 12))
        self.pause_button.pack(side=tk.LEFT, padx=10)

        self.resume_button = tk.Button(control_frame, text="4. Resume Smart Pauser", command=self.resume_script, state=tk.DISABLED, font=("Arial", 12))
        self.resume_button.pack(side=tk.LEFT, padx=10)

        self.kill_button = tk.Button(control_frame, text="5. End Smart Pauser", command=self.kill_script, state=tk.DISABLED, font=("Arial", 12))
        self.kill_button.pack(side=tk.LEFT, padx=10)

        self.resume_button = tk.Button(control_frame, text="Quit Application", command=self.stop_program, state=tk.DISABLED, font=("Arial", 12))
        self.resume_button.pack(side=tk.LEFT, padx=10)


        self.status_label = tk.Label(root, text="Status: Please calibrate first.", fg="red", font=("Arial", 10))
        self.status_label.pack(pady=5)

        self.text_output = ScrolledText(root, height=25, width=90, state="disabled", bg="black", fg="lightgray")
        self.text_output.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        self.root.protocol("WM_DELETE_WINDOW", self.stop_program)

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
        global proc
        self.log(f"[INFO] Starting Smart Pauser with center yaw {self.calibrated_yaw:.2f}...\n")
        self.start_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.resume_button.config(state=tk.DISABLED)
        self.text_output.config(state="normal")
        self.text_output.delete("1.0", tk.END)

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
    
    def pause_script(self):
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGSTOP)
            self.text_output.insert(tk.END, "\n[Process Paused]\n"); self.text_output.see(tk.END)
            # toggle buttons
            self.pause_button.config(state=tk.DISABLED)
            self.resume_button.config(state=tk.NORMAL)

    def resume_script(self):
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGCONT)
            self.text_output.insert(tk.END, "\n[Process Resumed]\n"); self.text_output.see(tk.END)
            # toggle buttons
            self.resume_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)

    def kill_script(self):
        if self.process and self.process.poll() is None:
            self.process.process.terminate()
            self.text_output.insert(tk.END, "\n[Process Terminated]\n"); self.text_output.see(tk.END)

    def stop_program(self):
        self.kill_script()
        root.destroy()


# def run_script():
#     global proc
#     start_button.config(state=tk.DISABLED)
#     pause_button.config(state=tk.NORMAL)
#     resume_button.config(state=tk.DISABLED)
#     self.text_output.config(state="normal")
#     text_output.delete("1.0", tk.END)

#     try:
#         proc = subprocess.Popen(
#             [sys.executable, SCRIPT],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             bufsize=1  # line-buffered
#         )
#         threading.Thread(target=reader_thread, args=(proc, out_q), daemon=True).start()
#         root.after(50, pump_output)
#     except Exception as e:
#         text_output.insert(tk.END, f"Error: {e}\n")
#         text_output.config(state="disabled")
#         start_button.config(state=tk.NORMAL)

# def pump_output():
#     """Non-blocking UI updater: drain the queue."""
#     if proc is None:
#         return
#     try:
#         while True:
#             line = out_q.get_nowait()
#             if line is None:
#                 # process finished
#                 rc = proc.poll()
#                 text_output.insert(tk.END, f"\n[Exited with code {rc}]\n")
#                 text_output.config(state="disabled")
#                 start_button.config(state=tk.NORMAL)
#                 return
#             text_output.insert(tk.END, line)
#             text_output.see(tk.END)
#     except queue.Empty:
#         pass
#     # keep polling
#     root.after(50, pump_output)

# def pause_script():
#     if proc and proc.poll() is None:
#         proc.send_signal(signal.SIGSTOP)
#         text_output.insert(tk.END, "\n[Process Paused]\n"); text_output.see(tk.END)
#         # toggle buttons
#         pause_button.config(state=tk.DISABLED)
#         resume_button.config(state=tk.NORMAL)

# def resume_script():
#     if proc and proc.poll() is None:
#         proc.send_signal(signal.SIGCONT)
#         text_output.insert(tk.END, "\n[Process Resumed]\n"); text_output.see(tk.END)
#         # toggle buttons
#         resume_button.config(state=tk.DISABLED)
#         pause_button.config(state=tk.NORMAL)

# def kill_script():
#     if proc and proc.poll() is None:
#         proc.terminate()  # or proc.kill()
#         text_output.insert(tk.END, "\n[Process Terminated]\n"); text_output.see(tk.END)

# def stop_program():
#     kill_script()
#     root.destroy()

# --- GUI setup ---
# root = tk.Tk()
# root.title("Subprocess Controller")
# root.geometry("600x400")

# start_button  = tk.Button(root, text="Run Script",   command=run_script);   start_button.pack(pady=5)
# pause_button  = tk.Button(root, text="Pause Script", command=pause_script); pause_button.pack(pady=5)
# resume_button = tk.Button(root, text="Resume Script",command=resume_script);resume_button.pack(pady=5)
# kill_button   = tk.Button(root, text="Kill Script",  command=kill_script);  kill_button.pack(pady=5)
# quit_button   = tk.Button(root, text="Quit",         command=stop_program); quit_button.pack(pady=5)

# text_output = ScrolledText(root, height=15, width=80, state="normal")
# text_output.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)



if __name__ == "__main__":
    root = tk.Tk()
    app = AppRunner(root)
    root.mainloop()
