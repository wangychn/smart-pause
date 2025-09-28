import requests
import json

# --- Configuration ---
# Define the server's IP address and port
IP_ADDRESS = '35.0.29.109'  # <-- The IP of your Flask server
PORT = 8000                 # <-- The port your Flask server is running on (e.g., 5000 or 8000)
ENDPOINT = '/get_data'      # <-- The specific route you want to request

# Construct the full URL for the request
URL = 'https://sicklily-legible-marline.ngrok-free.dev/get_data'

print("--- Flask HTTP Client ---")
print(f"Attempting to make a GET request to: {URL}")

try:
    # Make the GET request with a 5-second timeout
    response = requests.get(URL, timeout=5)

    # This line will automatically raise an error for bad status codes (4xx or 5xx)
    response.raise_for_status()

    # --- Success ---
    print(f"\nSuccess! Server responded with status code: {response.status_code}")

    # Try to parse the response as JSON (most common for APIs)
    try:
        data = response.json()
        print("Received JSON data:")
        # Pretty-print the JSON data
        print(json.dumps(data, indent=2))
    except json.JSONDecodeError:
        # If it's not JSON, print it as plain text
        print("Received non-JSON text data:")
        print(response.text)

# --- Error Handling ---
except requests.exceptions.ConnectionError as e:
    print("\nConnection Error: Failed to connect to the server.")
    print("Please check:")
    print("1. The IP address and port are correct.")
    print("2. The Flask server is running on the target machine.")
    print("3. There are no firewalls blocking the connection.")

except requests.exceptions.Timeout:
    print("\nRequest Timeout: The server did not respond within 5 seconds.")

except requests.exceptions.HTTPError as e:
    print(f"\nHTTP Error: The server returned a bad status code: {e.response.status_code}")
    print(f"Response Body: {e.response.text}")

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
