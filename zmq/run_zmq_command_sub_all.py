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

    # Send a request
    json_str = '{"command": "start", "value": 42}'
    ip_address = "localhost"
    DEFAULT_PORT_NUMBER = 5556
    port_number = DEFAULT_PORT_NUMBER

    # Socket to talk to server
    connection_url = f"tcp://{ip_address}:{port_number}"
    print(f"Connecting to publisher on {connection_url}…")

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(connection_url)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")  # Subscribe to all topics

    # Receive messages
    for _ in range(10):
        message = socket.recv()
        pmt_msg = pmt.deserialize_str(message)
        print(f"Received PMT message: {pmt.write_string(pmt_msg)}")
        time.sleep(1)  # Sleep to simulate processing time

    # Clean up
    socket.close()
    context.term()

if __name__ == '__main__':
    check_gnuradio_import()
    check_gtk()
    check_blocks_path()
    main()
