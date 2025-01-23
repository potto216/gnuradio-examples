#!/usr/bin/python3
"""
Diffie-Hellman Key Exchange Simulation Using ZeroMQ with Padding

This program simulates the Diffie-Hellman key exchange between Alice and Bob.
Messages are padded with a configurable number of zero bytes at the beginning
and end to simulate transmission over a GNU Radio RF channel.

Requirements:
- Simulate Alice and Bob in the same script using threads.
- Use ZMQ for message exchange with PUB/SUB sockets.
- Add configurable zero byte padding to messages.
- Detect the start and end of messages by stripping zero bytes.
"""

import sys
import threading
import time
import zmq
import logging
from datetime import datetime
from secrets import randbelow

# Constants for Diffie-Hellman
P = 23  # Prime number (public)
G = 5   # Primitive root modulo P (public)

def timestamp():
    """Returns the current time formatted as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class DiffieHellmanEntity:
    def __init__(self, name, pub_port, sub_port, zero_padding=0):
        self.name = name
        self.context = zmq.Context()
        
        # Set up PUB socket
        self.pub_socket = self.context.socket(zmq.PUB)
        try:
            self.pub_socket.bind(f"tcp://*:{pub_port}")
            logging.info(f"{self.name}: Successfully bound PUB socket to port {pub_port}")
        except zmq.ZMQError as e:
            logging.error(f"{self.name}: Failed to bind PUB socket to port {pub_port}: {e}")
        
        # Set up SUB socket
        self.sub_socket = self.context.socket(zmq.SUB)
        try:
            self.sub_socket.connect(f"tcp://localhost:{sub_port}")
            self.sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            logging.info(f"{self.name}: Successfully connected SUB socket to port {sub_port}")
        except zmq.ZMQError as e:
            logging.error(f"{self.name}: Failed to connect SUB socket to port {sub_port}: {e}")
        
        self.private_key = None
        self.public_key = None
        self.shared_secret = None
        self.zero_padding = zero_padding  # Number of zero bytes to add at beginning and end

    def generate_private_key(self):
        # Generate a private key (random integer)
        self.private_key = randbelow(P - 2) + 2  # Ensure private_key ∈ [2, P-1]
        logging.info(f"{self.name}: Generated private key: {self.private_key}")

    def compute_public_key(self):
        # Compute the public key: (G ** private_key) % P
        self.public_key = pow(G, self.private_key, P)
        logging.info(f"{self.name}: Computed public key: {self.public_key}")

    def publish_public_key(self):
        # Prepare the message
        message = f"PUBLIC_KEY {self.public_key}"
        # Encode the message to bytes
        message_bytes = message.encode('utf-8')
        # Add zero byte padding
        padded_message = b'\x00' * self.zero_padding + message_bytes + b'\x00' * self.zero_padding
        # Send the padded message
        self.pub_socket.send(padded_message)
        logging.info(f"{self.name}: Published public key with {self.zero_padding} zero bytes padding. Hex: {padded_message.hex()}")

    def receive_public_key(self):
        # Wait to receive the other party's public key
        while True:
            try:
                # Receive the message as bytes
                message_bytes = self.sub_socket.recv(flags=zmq.NOBLOCK)
                # Remove leading and trailing zero bytes
                message_stripped = message_bytes.strip(b'\x00')
                # Decode the message
                message = message_stripped.decode('utf-8')
                if message.startswith("PUBLIC_KEY"):
                    _, received_key = message.split()
                    received_key = int(received_key)
                    logging.info(f"{self.name}: Received public key: {received_key}. Hex: {message_bytes.hex()}")
                    return received_key
            except zmq.Again:
                # No message received yet
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"{self.name}: Error receiving public key: {e}")
                return None

    def compute_shared_secret(self, received_public_key):
        # Compute the shared secret: (received_public_key ** private_key) % P
        self.shared_secret = pow(received_public_key, self.private_key, P)
        logging.info(f"{self.name}: Computed shared secret: {self.shared_secret}")

    def close(self):
        # Cleanly close ZMQ sockets
        self.pub_socket.close()
        self.sub_socket.close()
        self.context.term()
        logging.info(f"{self.name}: Closed ZMQ sockets.")

def alice_thread(pub_port, sub_port, zero_padding):
    alice = DiffieHellmanEntity("Alice", pub_port=pub_port, sub_port=sub_port, zero_padding=zero_padding)
    try:
        alice.generate_private_key()
        alice.compute_public_key()
        time.sleep(1)  # Give time for sockets to connect
        alice.publish_public_key()
        bob_public_key = alice.receive_public_key()
        if bob_public_key is not None:
            alice.compute_shared_secret(bob_public_key)
    finally:
        alice.close()

def bob_thread(pub_port, sub_port, zero_padding):
    bob = DiffieHellmanEntity("Bob", pub_port=pub_port, sub_port=sub_port, zero_padding=zero_padding)
    try:
        bob.generate_private_key()
        bob.compute_public_key()
        time.sleep(1)  # Give time for sockets to connect
        bob.publish_public_key()
        alice_public_key = bob.receive_public_key()
        if alice_public_key is not None:
            bob.compute_shared_secret(alice_public_key)
    finally:
        bob.close()

def main(alice_pub_port, alice_sub_port, alice_zero_padding, bob_pub_port, bob_sub_port, bob_zero_padding):
    # Set up threads for Alice and Bob
    alice = threading.Thread(target=alice_thread, args=(alice_pub_port, alice_sub_port, alice_zero_padding))
    bob = threading.Thread(target=bob_thread, args=(bob_pub_port, bob_sub_port, bob_zero_padding))

    # Start threads
    alice.start()
    bob.start()

    # Wait for both threads to complete
    alice.join()
    bob.join()

    logging.info(f"Main: Diffie-Hellman key exchange simulation completed.")

if __name__ == "__main__":
    # Set up logging
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # logging.getLogger().handlers.clear()  # Clear the default handlers added by basicConfig

    file_handler = logging.FileHandler('diffie_hellman.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    
    # Example configuration
    # Internal test
    alice_pub_port = 5555
    alice_sub_port = 5556
    alice_zero_padding = 5
    bob_pub_port = 5556
    bob_sub_port = 5555
    bob_zero_padding = 3


    alice_pub_port = 5556
    alice_sub_port = 5557
    alice_zero_padding = 5000
    bob_pub_port = 5558
    bob_sub_port = 5559
    bob_zero_padding = 5000

    logging.info(f"Main: starting.")
    main(alice_pub_port, alice_sub_port, alice_zero_padding, bob_pub_port, bob_sub_port, bob_zero_padding)
    logging.info(f"Main: finished.")
else:
    logging.info(f"Main: did not run as a module.")

# python3 run_zmq_alice_bob.py > my_log.txt 2>&1
