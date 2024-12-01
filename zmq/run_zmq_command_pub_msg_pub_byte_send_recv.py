#!/usr/bin/python3
# Copyright 2016 Free Software Foundation, Inc.
# This file is part of GNU Radio
#
# SPDX-License-Identifier: GPL-2.0-or-later
#

import os
import sys
import zmq
import json
import time
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


def main():
    
    ip_address = "localhost"
    port_number_msg = 5555
    port_number_data = 5556
    port_number_sub = 5557
    data_length = 200  # Configurable amount of data
    max_loop = 500
    
    # Socket to talk to server
    connection_url_msg = f"tcp://{ip_address}:{port_number_msg}"
    connection_url_data = f"tcp://{ip_address}:{port_number_data}"
    connection_url_sub = f"tcp://{ip_address}:{port_number_sub}"
    print(f"Connecting to subscribers on {connection_url_msg}, {connection_url_data}, and {connection_url_sub}…")
    
    context = zmq.Context()
    socket_msg = context.socket(zmq.PUB)
    socket_data = context.socket(zmq.PUB)
    socket_sub = context.socket(zmq.SUB)
    socket_msg.bind(connection_url_msg)
    socket_data.bind(connection_url_data)
    socket_sub.connect(connection_url_sub)
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics
    
    amp_value = 1
    last_value = 0  # Initialize the last value
    
    # Open file to write received binary data as hex strings
    with open("received_data.txt", "w") as file:
        # Do max_loop requests, waiting each time for a response
        for request in range(max_loop):
            print(f"Sending request {request} …")
            
            pmt_msg = pmt.cons(pmt.intern("amp"), pmt.from_long(amp_value))
            
            # Send PMT message on socket 1
            print(f"Sending PMT message with amp: {amp_value}")
            socket_msg.send(pmt.serialize_str(pmt_msg))
            
            # Send counting sequence on socket 2        
            counting_sequence = bytes([(last_value + i) % 256 for i in range(data_length)])
            print(f"Sending counting sequence: {list(counting_sequence)}")
            socket_data.send(counting_sequence)
            
            # Update last_value and amp_value for the next iteration
            last_value = (last_value + data_length) % 256
            amp_value = (amp_value + 1) % 256
            
            # Receive binary data from socket_sub
            try:
                binary_data = socket_sub.recv(flags=zmq.NOBLOCK)
                hex_string = ' '.join(f'{byte:02x}' for byte in binary_data)
                print(f"Received binary data: {hex_string}")
                file.write(hex_string + '\n')
            except zmq.Again:
                # No message received
                pass
            
            time.sleep(1)  # Sleep to allow subscribers to process the message
    
    # Clean up
    socket_msg.close()
    socket_data.close()
    socket_sub.close()
    context.term()


if __name__ == '__main__':
    check_gnuradio_import()
    check_gtk()
    check_blocks_path()
    main()
