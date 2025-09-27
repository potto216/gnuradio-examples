#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: pkt_fsk_xmt_rcv_v1
# Author: Barry Duggan
# Description: packet FSK xmt rcv v1
# GNU Radio version: 3.10.9.2

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import audio
from gnuradio import blocks
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import math



class pkt_fsk_xmt_rcv_v1(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "pkt_fsk_xmt_rcv_v1", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("pkt_fsk_xmt_rcv_v1")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("GNU Radio", "pkt_fsk_xmt_rcv_v1")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)

        ##################################################
        # Variables
        ##################################################
        self.space = space = 3000
        self.mark = mark = 1000
        self.fsk_deviation = fsk_deviation = (abs)(mark-space)
        self.center = center = (mark+space)/2
        self.vco_max = vco_max = center+fsk_deviation
        self.samp_rate = samp_rate = 44100
        self.baud = baud = 200
        self.vco_offset = vco_offset = space/vco_max
        self.repeat = repeat = (int)(samp_rate/baud)
        self.decim = decim = 20
        self.thresh = thresh = 1
        self.sps = sps = (int)(repeat/decim)
        self.phase_bw = phase_bw = math.pi/32
        self.inp_amp = inp_amp = (mark/vco_max)-vco_offset

        ##################################################
        # Blocks
        ##################################################

        self.blocks_xor_xx_0 = blocks.xor_bb()
        self.blocks_wavfile_sink_0 = blocks.wavfile_sink(
            'pkt_fsk_xmt_make_wav_v1.wav',
            1,
            samp_rate,
            blocks.FORMAT_WAV,
            blocks.FORMAT_PCM_16,
            False
            )
        self.blocks_vector_source_x_0_0 = blocks.vector_source_b((0x01, 0x02, 0x03, 0x04, 0x05, 0x06), False, 1, [])
        self.blocks_vector_source_x_0 = blocks.vector_source_b((0x55, 0x55, 0x55), True, 1, [])
        self.blocks_vco_f_0 = blocks.vco_f(samp_rate, (2*math.pi*vco_max), 0.25)
        self.blocks_uchar_to_float_0 = blocks.uchar_to_float()
        self.blocks_repeat_0_0 = blocks.repeat(gr.sizeof_char*1, repeat)
        self.blocks_repeat_0 = blocks.repeat(gr.sizeof_char*1, (repeat*2))
        self.blocks_repack_bits_bb_1_0_0 = blocks.repack_bits_bb(8, 1, '', False, gr.GR_MSB_FIRST)
        self.blocks_repack_bits_bb_1_0 = blocks.repack_bits_bb(8, 1, '', False, gr.GR_MSB_FIRST)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff(inp_amp)
        self.blocks_add_const_vxx_0 = blocks.add_const_ff(vco_offset)
        self.audio_sink_0 = audio.sink(samp_rate, '', True)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.blocks_add_const_vxx_0, 0), (self.blocks_vco_f_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self.blocks_add_const_vxx_0, 0))
        self.connect((self.blocks_repack_bits_bb_1_0, 0), (self.blocks_repeat_0, 0))
        self.connect((self.blocks_repack_bits_bb_1_0_0, 0), (self.blocks_repeat_0_0, 0))
        self.connect((self.blocks_repeat_0, 0), (self.blocks_xor_xx_0, 0))
        self.connect((self.blocks_repeat_0_0, 0), (self.blocks_xor_xx_0, 1))
        self.connect((self.blocks_uchar_to_float_0, 0), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.blocks_vco_f_0, 0), (self.audio_sink_0, 0))
        self.connect((self.blocks_vco_f_0, 0), (self.blocks_wavfile_sink_0, 0))
        self.connect((self.blocks_vector_source_x_0, 0), (self.blocks_repack_bits_bb_1_0_0, 0))
        self.connect((self.blocks_vector_source_x_0_0, 0), (self.blocks_repack_bits_bb_1_0, 0))
        self.connect((self.blocks_xor_xx_0, 0), (self.blocks_uchar_to_float_0, 0))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("GNU Radio", "pkt_fsk_xmt_rcv_v1")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_space(self):
        return self.space

    def set_space(self, space):
        self.space = space
        self.set_center((self.mark+self.space)/2)
        self.set_fsk_deviation((abs)(self.mark-self.space))
        self.set_vco_offset(self.space/self.vco_max)

    def get_mark(self):
        return self.mark

    def set_mark(self, mark):
        self.mark = mark
        self.set_center((self.mark+self.space)/2)
        self.set_fsk_deviation((abs)(self.mark-self.space))
        self.set_inp_amp((self.mark/self.vco_max)-self.vco_offset)

    def get_fsk_deviation(self):
        return self.fsk_deviation

    def set_fsk_deviation(self, fsk_deviation):
        self.fsk_deviation = fsk_deviation
        self.set_vco_max(self.center+self.fsk_deviation)

    def get_center(self):
        return self.center

    def set_center(self, center):
        self.center = center
        self.set_vco_max(self.center+self.fsk_deviation)

    def get_vco_max(self):
        return self.vco_max

    def set_vco_max(self, vco_max):
        self.vco_max = vco_max
        self.set_inp_amp((self.mark/self.vco_max)-self.vco_offset)
        self.set_vco_offset(self.space/self.vco_max)

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate
        self.set_repeat((int)(self.samp_rate/self.baud))

    def get_baud(self):
        return self.baud

    def set_baud(self, baud):
        self.baud = baud
        self.set_repeat((int)(self.samp_rate/self.baud))

    def get_vco_offset(self):
        return self.vco_offset

    def set_vco_offset(self, vco_offset):
        self.vco_offset = vco_offset
        self.set_inp_amp((self.mark/self.vco_max)-self.vco_offset)
        self.blocks_add_const_vxx_0.set_k(self.vco_offset)

    def get_repeat(self):
        return self.repeat

    def set_repeat(self, repeat):
        self.repeat = repeat
        self.set_sps((int)(self.repeat/self.decim))
        self.blocks_repeat_0.set_interpolation((self.repeat*2))
        self.blocks_repeat_0_0.set_interpolation(self.repeat)

    def get_decim(self):
        return self.decim

    def set_decim(self, decim):
        self.decim = decim
        self.set_sps((int)(self.repeat/self.decim))

    def get_thresh(self):
        return self.thresh

    def set_thresh(self, thresh):
        self.thresh = thresh

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps

    def get_phase_bw(self):
        return self.phase_bw

    def set_phase_bw(self, phase_bw):
        self.phase_bw = phase_bw

    def get_inp_amp(self):
        return self.inp_amp

    def set_inp_amp(self, inp_amp):
        self.inp_amp = inp_amp
        self.blocks_multiply_const_vxx_0.set_k(self.inp_amp)




def main(top_block_cls=pkt_fsk_xmt_rcv_v1, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
