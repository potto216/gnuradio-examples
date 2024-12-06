#!/usr/bin/python3
"""
Diffie-Hellman Key Exchange Simulation Using ZeroMQ with Padding

This program simulates the Diffie-Hellman key exchange between multiple entities (e.g., Alice, Bob, Eve).
Messages are padded with a configurable number of zero bytes at the beginning and end to simulate transmission over a GNU Radio RF channel.

Requirements:
- Implement as a single flow without multiple threads, using a synchronous state machine.
- Synchronize actions of all entities.
- Make the program expandable to include additional actors like Eve.
- Use ZMQ for message exchange with PUB/SUB sockets connected to GNU Radio.
- Add configurable zero byte padding to messages.
- Detect the start and end of messages by stripping zero bytes.
"""

import os
import sys
import zmq
import json
import time
import logging
from datetime import datetime
from secrets import randbelow
import pmt


GR_IMPORT_ERROR_MESSAGE = """\
Cannot import gnuradio.

Is the Python path environment variable set correctly?
    All OS: PYTHONPATH

Is the library path environment variable set correctly?
    Linux: LD_LIBRARY_PATH
    Windows: PATH

See https://wiki.gnuradio.org/index.php/ModuleNotFoundError
"""
padding_byte = b'\x10'

def die(error, message):
    msg = "{0}\n\n({1})".format(message, error)
    try:
        import gi
        gi.require_version('Gtk', '3.0')
        from gi.repository import Gtk
        d = Gtk.MessageDialog(
            message_type=Gtk.MessageType.ERROR,
            buttons=Gtk.ButtonsType.CLOSE,
            text=msg,
        )
        d.set_title(type(error).__name__)
        d.run()
        sys.exit(1)
    except ImportError:
        sys.exit(type(error).__name__ + '\n\n' + msg)
    except Exception as _exception:
        print(
            "While trying to display an error message, another error occurred",
            file=sys.stderr)
        print(_exception, file=sys.stderr)
        print("The original error message follows.", file=sys.stderr)
        sys.exit(type(error).__name__ + '\n\n' + msg)


def check_gtk():
    try:
        import gi
        gi.require_version('Gtk', '3.0')
        gi.require_version('PangoCairo', '1.0')
        gi.require_foreign('cairo', 'Context')

        from gi.repository import Gtk
        success = Gtk.init_check()[0]
        if not success:
            # Don't display a warning dialogue. This seems to be a Gtk bug. If it
            # still can display warning dialogues, it does probably work!
            print(
                "Gtk init_check failed. GRC might not be able to start a GUI.",
                file=sys.stderr)

    except Exception as err:
        die(err, "Failed to initialize GTK. If you are running over ssh, "
                 "did you enable X forwarding and start ssh with -X?")


def check_gnuradio_import():
    try:
        from gnuradio import gr
    except ImportError as err:
        die(err, GR_IMPORT_ERROR_MESSAGE)


def check_blocks_path():
    if 'GR_DONT_LOAD_PREFS' in os.environ and not os.environ.get('GRC_BLOCKS_PATH', ''):
        die(EnvironmentError("No block definitions available"),
            "Can't find block definitions. Use config.conf or GRC_BLOCKS_PATH.")




# Constants for Diffie-Hellman
P = 23  # Prime number (public)
G = 5   # Primitive root modulo P (public)

def timestamp():
    """Returns the current time formatted as a string."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

class DiffieHellmanEntity:
    def __init__(self, name, pub_port, sub_ports, zero_padding=0):
        self.name = name
        self.context = zmq.Context()
        self.pub_socket = self.context.socket(zmq.PUB)
        try:
            self.pub_socket.bind(f"tcp://*:{pub_port}")
            logging.info(f"{self.name}: Successfully bound PUB socket to port {pub_port}")
        except zmq.ZMQError as e:
            logging.error(f"{self.name}: Failed to bind PUB socket to port {pub_port}: {e}")
            sys.exit(1)
        
        self.sub_sockets = []
        for sub_port in sub_ports:
            sub_socket = self.context.socket(zmq.SUB)
            try:
                sub_socket.connect(f"tcp://localhost:{sub_port}")
                sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
                self.sub_sockets.append(sub_socket)
                logging.info(f"{self.name}: Successfully connected SUB socket to port {sub_port}")
            except zmq.ZMQError as e:
                logging.error(f"{self.name}: Failed to connect SUB socket to port {sub_port}: {e}")
                sys.exit(1)
        
        self.private_key = None
        self.public_key = None
        self.shared_secrets = {}  # Dictionary to store shared secrets with other entities
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
        message = f"PUBLIC_KEY {self.name} {self.public_key}"
        # Encode the message to bytes
        message_bytes = message.encode('utf-8')
        # Add zero byte padding
        padded_message = padding_byte * self.zero_padding + message_bytes + padding_byte * self.zero_padding
        # Send the padded message
        self.pub_socket.send(padded_message)
        logging.info(f"{self.name}: Published public key with {self.zero_padding} zero bytes padding. The message was {message}")

    def receive_public_keys(self, expected_entities):
        # Collect public keys from other entities
        received_keys = {}
        start_time = time.time()
        timeout = 5  # Seconds to wait for messages
        while len(received_keys) < len(expected_entities):
            for sub_socket in self.sub_sockets:
                try:
                    # Receive the message as bytes
                    message_bytes = sub_socket.recv(flags=zmq.NOBLOCK)
                    # Remove leading and trailing zero bytes
                    message_stripped = message_bytes.strip(padding_byte)
                    # Decode the message
                    message = message_stripped.decode('utf-8')
                    if message.startswith("PUBLIC_KEY"):
                        _, sender_name, received_key = message.split()
                        if sender_name != self.name and sender_name in expected_entities:
                            received_key = int(received_key)
                            received_keys[sender_name] = received_key
                            logging.info(f"{self.name}: Received public key from {sender_name}: {received_key}")
                except zmq.Again:
                    # No message received yet
                    pass
                except Exception as e:
                    logging.error(f"{self.name}: Error receiving public key: {e}")
                    continue
            if time.time() - start_time > timeout:
                logging.error(f"{self.name}: Timeout while waiting for public keys.")
                break
            time.sleep(0.1)
        return received_keys

    def compute_shared_secrets(self, received_keys):
        for sender_name, received_key in received_keys.items():
            shared_secret = pow(received_key, self.private_key, P)
            self.shared_secrets[sender_name] = shared_secret
            logging.info(f"{self.name}: Computed shared secret with {sender_name}: {shared_secret}")

    def close(self):
        # Cleanly close ZMQ sockets
        self.pub_socket.close()
        for sub_socket in self.sub_sockets:
            sub_socket.close()
        self.context.term()
        logging.info(f"{self.name}: Closed ZMQ sockets.")

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('diffie_hellman.log', mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logger = logging.getLogger()
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    
    # Configure entities
    # For simplicity, we'll use ports 5555 to 5559
    zero_padding = 2000
    entities_info = [
        {
            "name": "Alice",
            "pub_port": 5556, # TX 
            "sub_ports": [5557], # RX
            "zero_padding": zero_padding # RX
        },
        {
            "name": "Bob",
            "pub_port": 5558, # TX 
            "sub_ports": [5559], # RX
            "zero_padding": zero_padding
        },
        {
            "name": "Eve",
            "pub_port": 5560, # TX 
            "sub_ports": [5561], # RX
            "zero_padding": zero_padding
        }
    ]

    # Create entities
    entities = []
    for info in entities_info:
        entity = DiffieHellmanEntity(
            name=info["name"],
            pub_port=info["pub_port"],
            sub_ports=info["sub_ports"],
            zero_padding=info["zero_padding"]
        )
        entities.append(entity)

    # List of entity names for reference
    entity_names = [entity.name for entity in entities]

    # Step 1: Each entity generates its private key
    for entity in entities:
        entity.generate_private_key()

    # Step 2: Each entity computes its public key
    for entity in entities:
        entity.compute_public_key()

    # Allow time for sockets to connect
    time.sleep(1)

    # Step 3: Each entity publishes its public key
    for entity in entities:
        entity.publish_public_key()

    # Allow time for messages to be sent
    time.sleep(1)

    # Step 4: Each entity receives public keys from others
    for entity in entities:
        expected_entities = [name for name in entity_names if name != entity.name]
        received_keys = entity.receive_public_keys(expected_entities)
        entity.compute_shared_secrets(received_keys)

    # Step 5: Verify shared secrets
    for entity in entities:
        logging.info(f"{entity.name}: Shared secrets with others:")
        for other_name, secret in entity.shared_secrets.items():
            logging.info(f"  - {other_name}: {secret}")

    # Close all entities
    for entity in entities:
        entity.close()

    logging.info("Main: Diffie-Hellman key exchange simulation completed.")


if __name__ == '__main__':
    check_gnuradio_import()
    check_gtk()
    check_blocks_path()
    main()
