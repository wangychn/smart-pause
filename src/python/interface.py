import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import subprocess
import sys
import os
import threading
import socket
import json
import time
from collections import deque
import signal

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# IMPORTANT: This now points to the new socket-based CLIENT script
PAUSE_SCRIPT = os.path.join(SCRIPT_DIR, "test_server.py")
CALIBRATE_SCRIPT = os.path.join(SCRIPT_DIR, "calibrate.py")
SOCKET_HOST = "0.0.0.0"
SOCKET_PORT = 65432


class AppRunner:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Pauser Control (Server Mode)")
        self.root.geometry("700x550")
        self.return_data = ""
        self.calibrated_yaw = None
        self.process: subprocess.Popen | None = None
        
        # --- Socket Server State ---
        self.server_socket: socket.socket | None = None
        self.client_connection: socket.socket | None = None

        self.live_stats = {}  # <-- Stores the latest stats
        self.live_stats_lines = deque(maxlen=5)

        # --- Main layout containers ---
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        sidebar = tk.Frame(main_frame, padx=8, pady=8)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        content = tk.Frame(main_frame)
        content.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Buttons ---
        self.calibrate_button = tk.Button(sidebar, text="1. Calibrate", command=self.run_calibration, font=("Arial", 12))
        self.calibrate_button.pack(fill=tk.X, pady=4)
        self.start_button = tk.Button(sidebar, text="2. Run", command=self.run_pauser, state=tk.DISABLED, font=("Arial", 12))
        self.start_button.pack(fill=tk.X, pady=4)
        self.pause_button = tk.Button(sidebar, text="3. Pause", command=self.pause_script, state=tk.DISABLED, font=("Arial", 12))
        self.pause_button.pack(fill=tk.X, pady=4)
        self.resume_button = tk.Button(sidebar, text="4. Resume", command=self.resume_script, state=tk.DISABLED, font=("Arial", 12))
        self.resume_button.pack(fill=tk.X, pady=4)
        self.kill_button = tk.Button(sidebar, text="5. End", command=self.kill_script, state=tk.DISABLED, font=("Arial", 12))
        self.kill_button.pack(fill=tk.X, pady=4)
        self.quit_button = tk.Button(sidebar, text="Quit", command=self.stop_program, state=tk.NORMAL, font=("Arial", 12))
        self.quit_button.pack(fill=tk.X, pady=12)

        # --- Status + output ---
        self.status_label = tk.Label(content, text="Status: Please calibrate first.", fg="red", font=("Arial", 10))
        self.status_label.pack(pady=(10, 4), padx=10, anchor="w")
        stats_frame = tk.LabelFrame(content, text="Live Stats", padx=5, pady=5, font=("Arial", 10, "bold"))
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        self.stats_display = tk.Label(stats_frame, text="Waiting for pauser to start...", font=("Courier New", 10), justify=tk.LEFT, anchor="nw")
        self.stats_display.pack(fill=tk.X)
        self.text_output = ScrolledText(content, height=25, width=90, state="disabled", bg="black", fg="lightgray", font=("Courier New", 9))
        self.text_output.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self.stop_program)

    # -------- helpers ----------
    def log(self, message: str):
        self.text_output.config(state="normal")
        self.text_output.insert(tk.END, message)
        self.text_output.see(tk.END)
        self.text_output.config(state="disabled")

    def _reset_buttons_to_idle(self):
        self.start_button.config(state=tk.NORMAL if self.calibrated_yaw is not None else tk.DISABLED)
        self.calibrate_button.config(state=tk.NORMAL)
        self.pause_button.config(state=tk.DISABLED)
        self.resume_button.config(state=tk.DISABLED)
        self.kill_button.config(state=tk.DISABLED)

    def _set_running_buttons(self):
        self.start_button.config(state=tk.DISABLED)
        self.calibrate_button.config(state=tk.DISABLED)
        self.pause_button.config(state=tk.NORMAL)
        self.resume_button.config(state=tk.DISABLED)
        self.kill_button.config(state=tk.NORMAL)

    # -------- calibration (unchanged) ----------
    def run_calibration(self):
        self.log("[INFO] Starting calibration...\n")
        self.calibrate_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Calibration in progress...", fg="orange")
        try:
            self.process = subprocess.Popen([sys.executable, CALIBRATE_SCRIPT], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            threading.Thread(target=self._wait_for_calibration, daemon=True).start()
        except Exception as e:
            self.log(f"[ERROR] Failed to launch calibration: {e}\n")
            self.calibrate_button.config(state=tk.NORMAL)

    def _wait_for_calibration(self):
        stdout, _ = self.process.communicate()
        self.root.after(0, self._handle_calibration_result, stdout, self.process.returncode)

    def _handle_calibration_result(self, stdout: str, returncode: int):
        if returncode == 0:
            try:
                self.calibrated_yaw = float(stdout.strip().splitlines()[-1])
                self.log(f"[SUCCESS] Calibrated. Center Yaw: {self.calibrated_yaw:.2f}\n")
                self.status_label.config(text=f"Status: Calibrated. Ready.", fg="green")
                self.start_button.config(state=tk.NORMAL)
            except (ValueError, IndexError):
                self.log("[ERROR] Could not parse calibration value.\n")
                self.status_label.config(text="Status: Calibration failed.", fg="red")
        else:
            self.log("[ERROR] Calibration script failed.\n")
        self.calibrate_button.config(state=tk.NORMAL)
        self.process = None

    # -------- runner (socket server implementation) ----------
    def run_pauser(self):
        if self.calibrated_yaw is None: return
        self._set_running_buttons()
        self.text_output.config(state="normal"); self.text_output.delete("1.0", tk.END); self.text_output.config(state="disabled")
        self.live_stats_lines.clear(); self.stats_display.config(text="")
        
        # Start server listening thread, which will then launch the subprocess
        threading.Thread(target=self.start_server_and_launch_client, daemon=True).start()

    def start_server_and_launch_client(self):
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((SOCKET_HOST, SOCKET_PORT))
            self.server_socket.listen()
            self.root.after(0, self.log, f"[INFO] Server listening on {SOCKET_HOST}:{SOCKET_PORT}\n")
            self.root.after(0, self.status_label.config, {"text": "Status: Waiting for client...", "fg": "orange"})

            # --- Launch the client subprocess AFTER the server is listening ---
            self.process = subprocess.Popen([sys.executable, PAUSE_SCRIPT, str(self.calibrated_yaw), SOCKET_HOST, str(SOCKET_PORT)])
            
            # --- Wait for the client to connect ---
            self.client_connection, addr = self.server_socket.accept()
            self.root.after(0, self.log, f"[INFO] Client connected from {addr}\n")
            self.root.after(0, self.status_label.config, {"text": "Status: Running", "fg": "blue"})
            
            # Start listening for messages from the client
            threading.Thread(target=self.listen_for_messages, daemon=True).start()

        except Exception as e:
            self.root.after(0, self.log, f"[ERROR] Server/Client launch failed: {e}\n")
            self.root.after(0, self._on_process_exit)

    # ==============================================================================
    # == THIS IS THE UPDATED METHOD THAT PARSES YOUR SPECIFIC STRING FORMAT ==
    # ==============================================================================
    def listen_for_messages(self):
        buffer = ""
        try:
            while self.client_connection:
                data = self.client_connection.recv(1024).decode('utf-8')
                if not data: break
                
                buffer += data
                while '\n' in buffer:
                    message, buffer = buffer.split('\n', 1)
                    message = message.strip()
                    if not message: continue

                    try:
                        stats_dict = {}
                        # Split the message by commas to get each key-value pair
                        parts = message.split(',')
                        
                        for part in parts:
                            part = part.strip()
                            if not part: continue

                            # The server will try to split by ':' first
                            if ':' in part:
                                key, value = part.split(':', 1)
                            # If no ':', it will handle the "Key Key Value" format
                            # by splitting just once from the right on the space
                            else:
                                try:
                                    key, value = part.rsplit(' ', 1)
                                except ValueError:
                                    # This will skip any parts that don't have a space (e.g., just "Error")
                                    continue
                            
                            stats_dict[key.strip()] = value.strip()
                        
                        if stats_dict:
                            self.live_stats = stats_dict
                            self.root.after(0, self.update_live_output)
                        else:
                             # If nothing could be parsed, log the raw message
                            self.root.after(0, self.log, f"Unparsed message: {message}\n")

                    except Exception as e:
                        self.root.after(0, self.log, f"[PARSE ERROR] for '{message}': {e}\n")

        except (ConnectionResetError, OSError):
             self.log("[INFO] Client connection closed.\n")
        finally:
            self.root.after(0, self._on_process_exit)

    def update_live_output(self):
        """Updates the stats display using data from self.live_stats."""
        self.live_stats_lines.clear()
        # Read directly from the class attribute
        for key, value in self.live_stats.items():
            self.live_stats_lines.append(f"{key:<20}: {value}")
        self.stats_display.config(text="\n".join(self.live_stats_lines))
            
    def _on_process_exit(self):
        self.log("\n[INFO] Pauser script has exited or is being terminated.\n")
        self.status_label.config(text="Status: Not running", fg="black")
        
        # Close network connections first
        if self.client_connection: self.client_connection.close(); self.client_connection = None
        if self.server_socket: self.server_socket.close();   self.server_socket = None
        
        # --- New logic to terminate ONLY the subprocess ---
        if self.process and self.process.poll() is None:
            self.log("[INFO] Attempting to terminate the pause.py subprocess...\n")
            self.process.terminate()
            try:
                self.process.communicate(timeout=1.0)
                self.log("[INFO] Subprocess terminated gracefully.\n")
            except subprocess.TimeoutExpired:
                self.log("[WARN] Subprocess did not respond. Forcing kill...\n")
                self.process.kill()
                self.process.communicate()

        # Reset state
        self.process = None
        self._reset_buttons_to_idle()

    # -------- controls (socket implementation) ----------
    def send_command(self, command: str):
        if not self.client_connection: return
        try:
            self.client_connection.sendall((json.dumps({"command": command}) + '\n').encode('utf-8'))
        except Exception as e:
            self.log(f"[ERROR] Failed to send command '{command}': {e}\n")
    
    def pause_script(self):
        if not self.process or self.process.poll() is not None:
            return
        try:
            self.process.send_signal(signal.SIGSTOP)
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
            self.log("[INFO] Resumed.\n")
            self.resume_button.config(state=tk.DISABLED)
            self.pause_button.config(state=tk.NORMAL)
        except Exception as e:
            self.log(f"[ERROR] Resume failed: {e}\n")

    def kill_script(self):
        """Initiates a shutdown of the client process and cleans up all resources."""
        self.status_label.config(text="Status: Terminating...", fg="red")
        
        if not self.process and not self.client_connection:
            self.log("[INFO] Kill command ignored: No process is active.\n")
            self.status_label.config(text="Status: Not running", fg="black")
            return

        self.log("[INFO] Displaying final stats before shutdown...\n")
        self.update_live_output()
        self.root.update_idletasks() # Force the GUI to redraw immediately.

        self.log("[INFO] Sending shutdown command to client...\n")
        self.send_command("shutdown")
        self._on_process_exit()

    def stop_program(self):
        try:
            if self.process: self.kill_script()
            time.sleep(0.1) # Give socket time to close
        finally:
            self.root.destroy()        

if __name__ == "__main__":
    root = tk.Tk()
    app = AppRunner(root)
    root.mainloop()