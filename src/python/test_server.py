# web_server.py
from flask import Flask, jsonify
import random

# Create a Flask web server application
app = Flask(__name__)

# This is your "API endpoint". When the client makes a request to '/get_data',
# this function will run and return a response.
@app.route("/get_data")
def get_some_data():
    """
    Generates some data to send back to the client.
    In the original code, you were expecting a number, so we'll simulate that.
    """
    # For this example, we'll just send a random number.
    # You can replace this with whatever logic you need.
    some_number = random.randint(1, 100)
    
    # Send the data back in a standard JSON format
    response_data = {"number": some_number}
    return jsonify(response_data)

if __name__ == '__main__':
    print("Starting Flask web server...")
    # Run the server on port 8000, accessible from any IP address.
    # Make sure port 8000 is open in your firewall/security group.
    app.run(host='0.0.0.0', port=8000)