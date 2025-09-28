# test_client.py
import socket

# The server's IP address
HOST = '172.29.112.216' # <-- Make sure this IP is correct!
PORT = 65432

print("--- Client ---")
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f"Attempting to connect to {HOST}:{PORT}...")
        s.connect((HOST, PORT))
        data = s.recv(1024)
        print(f"Success! Received: {data.decode('utf-8')}")

except ConnectionRefusedError:
    print("Connection failed: The server actively refused it.")
except TimeoutError:
    print("Connection failed: The request timed out. Check firewall or network issues.")
except Exception as e:
    print(f"An error occurred: {e}")