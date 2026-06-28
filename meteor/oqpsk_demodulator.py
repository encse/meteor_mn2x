# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: OQPSK Demodulator
# GNU Radio version: 3.10.12.0

from gnuradio import analog
from gnuradio import blocks
from gnuradio import digital
from gnuradio import filter
from gnuradio.filter import firdes
from gnuradio import gr
from gnuradio.fft import window
import sys
import signal
import math
import tag_to_float
import threading







class oqpsk_demodulator(gr.hier_block2):
    def __init__(self, sample_rate=0):
        gr.hier_block2.__init__(
            self, "OQPSK Demodulator",
                gr.io_signature(1, 1, gr.sizeof_gr_complex*1),
                gr.io_signature.makev(3, 3, [gr.sizeof_gr_complex*1, gr.sizeof_float*1, gr.sizeof_float*1]),
        )

        ##################################################
        # Parameters
        ##################################################
        self.sample_rate = sample_rate

        ##################################################
        # Variables
        ##################################################
        self.sym_rate = sym_rate = 72000
        self.sps = sps = 2
        self.rrc_alpha = rrc_alpha = 0.5
        self.pipeline_sample_rate = pipeline_sample_rate = 144000

        ##################################################
        # Blocks
        ##################################################

        self.tag_value_to_float_0 = tag_to_float.TagToFloat(tag_key='snr')
        self.rational_resampler_xxx_0 = filter.rational_resampler_ccc(
                interpolation=pipeline_sample_rate,
                decimation=sample_rate,
                taps=[],
                fractional_bw=0)
        self.fir_filter_xxx_1 = filter.fir_filter_ccc(1, firdes.root_raised_cosine(1.0, pipeline_sample_rate, sym_rate, alpha=rrc_alpha, ntaps=35))
        self.fir_filter_xxx_1.declare_sample_delay(0)
        self.digital_symbol_sync_xx_0 = digital.symbol_sync_cc(
            digital.TED_GARDNER,
            sps,
            0.0087,
            0.707,
            1,
            0.01,
            1,
            digital.constellation_qpsk().base(),
            digital.IR_MMSE_8TAP,
            128,
            [])
        self.digital_mpsk_snr_est_cc_0 = digital.mpsk_snr_est_cc(2, 10000, 0.001)
        self.digital_costas_loop_cc_0 = digital.costas_loop_cc(0.002, 4, False)
        self.blocks_null_sink_0_0 = blocks.null_sink(gr.sizeof_float*1)
        self.blocks_null_sink_0 = blocks.null_sink(gr.sizeof_float*1)
        self.blocks_multiply_const_vxx_0 = blocks.multiply_const_ff((1.0 / (2.0 * math.pi) * pipeline_sample_rate))
        self.blocks_float_to_complex_0_0 = blocks.float_to_complex(1)
        self.blocks_delay_0_0 = blocks.delay(gr.sizeof_float*1, (sps // 2))
        self.blocks_complex_to_float_0_0 = blocks.complex_to_float(1)
        self.analog_agc_xx_0 = analog.agc_cc((1e-3), 1.0, 1.0, 65536)


        ##################################################
        # Connections
        ##################################################
        self.connect((self.analog_agc_xx_0, 0), (self.fir_filter_xxx_1, 0))
        self.connect((self.blocks_complex_to_float_0_0, 1), (self.blocks_delay_0_0, 0))
        self.connect((self.blocks_complex_to_float_0_0, 0), (self.blocks_float_to_complex_0_0, 0))
        self.connect((self.blocks_delay_0_0, 0), (self.blocks_float_to_complex_0_0, 1))
        self.connect((self.blocks_float_to_complex_0_0, 0), (self.digital_symbol_sync_xx_0, 0))
        self.connect((self.blocks_multiply_const_vxx_0, 0), (self, 1))
        self.connect((self.digital_costas_loop_cc_0, 0), (self.blocks_complex_to_float_0_0, 0))
        self.connect((self.digital_costas_loop_cc_0, 1), (self.blocks_multiply_const_vxx_0, 0))
        self.connect((self.digital_costas_loop_cc_0, 3), (self.blocks_null_sink_0, 0))
        self.connect((self.digital_costas_loop_cc_0, 2), (self.blocks_null_sink_0_0, 0))
        self.connect((self.digital_mpsk_snr_est_cc_0, 0), (self.tag_value_to_float_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self.digital_mpsk_snr_est_cc_0, 0))
        self.connect((self.digital_symbol_sync_xx_0, 0), (self, 0))
        self.connect((self.fir_filter_xxx_1, 0), (self.digital_costas_loop_cc_0, 0))
        self.connect((self, 0), (self.rational_resampler_xxx_0, 0))
        self.connect((self.rational_resampler_xxx_0, 0), (self.analog_agc_xx_0, 0))
        self.connect((self.tag_value_to_float_0, 0), (self, 2))


    def get_sample_rate(self):
        return self.sample_rate

    def set_sample_rate(self, sample_rate):
        self.sample_rate = sample_rate

    def get_sym_rate(self):
        return self.sym_rate

    def set_sym_rate(self, sym_rate):
        self.sym_rate = sym_rate
        self.fir_filter_xxx_1.set_taps(firdes.root_raised_cosine(1.0, self.pipeline_sample_rate, self.sym_rate, alpha=self.rrc_alpha, ntaps=35))

    def get_sps(self):
        return self.sps

    def set_sps(self, sps):
        self.sps = sps
        self.blocks_delay_0_0.set_dly(int((self.sps // 2)))
        self.digital_symbol_sync_xx_0.set_sps(self.sps)

    def get_rrc_alpha(self):
        return self.rrc_alpha

    def set_rrc_alpha(self, rrc_alpha):
        self.rrc_alpha = rrc_alpha
        self.fir_filter_xxx_1.set_taps(firdes.root_raised_cosine(1.0, self.pipeline_sample_rate, self.sym_rate, alpha=self.rrc_alpha, ntaps=35))

    def get_pipeline_sample_rate(self):
        return self.pipeline_sample_rate

    def set_pipeline_sample_rate(self, pipeline_sample_rate):
        self.pipeline_sample_rate = pipeline_sample_rate
        self.blocks_multiply_const_vxx_0.set_k((1.0 / (2.0 * math.pi) * self.pipeline_sample_rate))
        self.fir_filter_xxx_1.set_taps(firdes.root_raised_cosine(1.0, self.pipeline_sample_rate, self.sym_rate, alpha=self.rrc_alpha, ntaps=35))

