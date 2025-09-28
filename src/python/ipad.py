import socket
import random
import time
import sys

# --- Configuration ---
# Use '0.0.0.0' to listen on all available network interfaces.
# This is necessary for other computers on the network to connect.
HOST = '0.0.0.0' 
PORT = 65432  # An arbitrary non-privileged port

# --- Main Server Logic ---
print("--- Number Sender Server ---")
print("Starting server...")

# socket.AF_INET specifies the IPv4 address family.
# socket.SOCK_STREAM specifies the TCP connection type.
# The 'with' statement ensures the socket is automatically closed.
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
    # Bind the socket to the address and port
    server_socket.bind((HOST, PORT))
    
    # Enable the server to accept connections, with a backlog of 1
    server_socket.listen(1)
    
    print(f"Server is listening on port {PORT}...")
    print("Waiting for a client to connect...")
    
    # Accept a connection. This is a blocking call.
    # It waits until a client connects.
    # 'conn' is a new socket object usable to send and receive data on the connection.
    # 'addr' is the address bound to the socket on the other end of the connection.
    conn, addr = server_socket.accept()
    
    with conn:
        print(f"Connected by {addr}")
        print("Starting to send random numbers (0 or 1) every 2 seconds.")
        print("Press Ctrl+C to stop.")
        
        try:
            while True:
                # Generate a random number: 0 or 1
                random_number = random.randint(0, 1)
                
                # Convert the number to a string and then to bytes for sending
                message = str(random_number).encode('utf-8')
                
                # Send the data to the connected client
                conn.sendall(message)
                
                print(f"Sent: {random_number}")
                
                # Wait for 2 seconds before sending the next number
                time.sleep(2)

        except (ConnectionResetError, BrokenPipeError):
            print(f"\nClient {addr} disconnected.")
        except KeyboardInterrupt:
            print("\nServer is shutting down.")
        finally:
            print("Connection closed.")
