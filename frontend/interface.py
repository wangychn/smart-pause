import tkinter as tk
from tkinter.scrolledtext import ScrolledText
import subprocess
import sys
import os

# Path to the script you want to run
SCRIPT = os.path.join(os.path.dirname(__file__), "pause.py")

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