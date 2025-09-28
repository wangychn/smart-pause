# test_server.py
import socket

# Listen on all available network interfaces
HOST = '0.0.0.0'
PORT = 65432

print("--- Server ---")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print(f"Waiting for a connection on port {PORT}...")
    
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        # Send a message to the client
        conn.sendall(b'Hello from the server!')