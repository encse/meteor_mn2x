import ctypes
import numpy as np
from gnuradio import gr, fec


def ndarray_to_capsule(arr):
    """
    Wrap a contiguous numpy array data pointer into a PyCapsule.
    The numpy array must stay alive while the capsule is used.
    """
    if not arr.flags["C_CONTIGUOUS"]:
        raise ValueError("Array must be C-contiguous")

    ptr = ctypes.c_void_p(arr.ctypes.data)

    pycapsule_new = ctypes.pythonapi.PyCapsule_New
    pycapsule_new.restype = ctypes.py_object
    pycapsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]

    return pycapsule_new(ptr, None, None)


class Viterbi(gr.basic_block):
    """
    Convolutional Viterbi decoder block for complex OQPSK/QPSK soft symbols.

    Input:
        complex64 stream

    Each complex sample contains two soft symbols:

        real -> first soft symbol
        imag -> second soft symbol

    Normal interpretation:

        z0 = I0 + jQ0
        z1 = I1 + jQ1
        z2 = I2 + jQ2

        soft stream:
            I0, Q0, I1, Q1, I2, Q2, ...

    If the OQPSK branch compensation delayed Q in the wrong direction,
    the symbol sync output may instead look like:

        z0 = I0 + jQ-1
        z1 = I1 + jQ0
        z2 = I2 + jQ1

    In that case we need Q lookahead:

        corrected0 = I0 + jQ0 = real(z0) + j imag(z1)
        corrected1 = I1 + jQ1 = real(z1) + j imag(z2)

    This block tries four candidates:

        A: normal,      0 degree rotation
        B: normal,      90 degree rotation
        C: q_lookahead, 0 degree rotation
        D: q_lookahead, 90 degree rotation

    Important:
        For q_lookahead, we first rebuild the corrected complex symbols
        and only then apply the 90 degree rotation. Do not rotate the raw
        IQ stream before doing q_lookahead.
    """

    BLOCK_BITS = 4096
    BLOCK_SOFT = BLOCK_BITS * 2

    def __init__(self):
        gr.basic_block.__init__(
            self,
            name="viterbi",
            in_sig=[np.complex64],
            out_sig=[np.uint8, np.float32],
        )

        polys = [109, 79]

        self.dec = fec.cc_decoder.make(
            self.BLOCK_BITS,
            7,
            2,
            polys,
            0,
            -1,
            fec.CC_STREAMING,
            False,
        )

        self.enc = fec.cc_encoder.make(
            self.BLOCK_BITS,
            7,
            2,
            polys,
            0,
            fec.CC_STREAMING,
            False,
        )

        self.history_overlap = int(self.dec.get_history())

        if self.history_overlap < 0 or self.history_overlap % 2 != 0:
            raise RuntimeError("Decoder history is invalid")

        # One complex IQ sample contains two soft symbols.
        self.history_iq = self.history_overlap // 2

        # Previous input samples needed by the GNU Radio FEC decoder history.
        self.prev_iq = np.zeros(self.history_iq, dtype=np.complex64)

        # full_iq layout:
        #
        #   [ history_iq previous samples ][ BLOCK_BITS current samples ][ 1 lookahead sample ]
        #
        # The extra lookahead sample is needed for q_lookahead.
        self.full_iq = np.zeros(
            self.history_iq + self.BLOCK_BITS + 1,
            dtype=np.complex64,
        )

        # Viterbi input contains history_overlap soft values plus current block soft values.
        self.soft_float = np.zeros(
            self.history_overlap + self.BLOCK_SOFT,
            dtype=np.float32,
        )

        self.soft_u8 = np.zeros(
            self.history_overlap + self.BLOCK_SOFT,
            dtype=np.uint8,
        )

        # Viterbi decoded output includes history_iq decoded bits plus current block bits.
        self.decoded_tmp = np.zeros(
            self.history_iq + self.BLOCK_BITS,
            dtype=np.uint8,
        )

        self.decoded_best = np.zeros(
            self.BLOCK_BITS,
            dtype=np.uint8,
        )

        self.reencoded = np.zeros(
            self.history_overlap + self.BLOCK_SOFT,
            dtype=np.uint8,
        )

        self.soft_caps = ndarray_to_capsule(self.soft_u8)
        self.decoded_tmp_caps = ndarray_to_capsule(self.decoded_tmp)
        self.renc_caps = ndarray_to_capsule(self.reencoded)

        self.candidates = [
            ("normal_rot0", "normal", 0),
            ("normal_rot90", "normal", 90),
            ("q_lookahead_rot0", "q_lookahead", 0),
            ("q_lookahead_rot90", "q_lookahead", 90),
        ]

        self.last_candidate = "none"
        self.last_ber = 10.0

    def forecast(self, noutput_items, ninputs):
        # +1 because q_lookahead uses imag of the next complex sample.
        return [self.BLOCK_BITS + 1]

    def build_full_iq(self, iq_window):
        """
        iq_window must contain BLOCK_BITS + 1 samples.

        We consume only BLOCK_BITS samples. The final sample is only lookahead.
        """
        if self.history_iq > 0:
            self.full_iq[:self.history_iq] = self.prev_iq

        self.full_iq[self.history_iq:] = iq_window[: self.BLOCK_BITS + 1]

    def make_symbols(self, start, count, mode):
        """
        Build corrected complex symbols from full_iq.

        mode == "normal":
            z[n] = I[n] + jQ[n]

        mode == "q_lookahead":
            z[n] = I[n] + jQ[n+1]
        """
        end = start + count

        if mode == "normal":
            i = self.full_iq[start:end].real
            q = self.full_iq[start:end].imag

        elif mode == "q_lookahead":
            i = self.full_iq[start:end].real
            q = self.full_iq[start + 1 : end + 1].imag

        else:
            raise ValueError(f"unknown mode: {mode}")

        if len(i) != count:
            raise RuntimeError("Internal error: invalid I slice length")

        if len(q) != count:
            raise RuntimeError("Internal error: invalid Q slice length")

        return i.astype(np.float32, copy=False) + np.complex64(1j) * q.astype(
            np.float32,
            copy=False,
        )

    def apply_rotation(self, z, rotation):
        """
        Rotate already-corrected complex symbols.

        Important: rotation is applied after q_lookahead pairing.
        """
        if rotation == 0:
            return z

        if rotation == 90:
            return z * np.complex64(1j)

        raise ValueError(f"unknown rotation: {rotation}")

    def build_soft_input(self, mode, rotation):
        """
        Build scalar soft input for GNU Radio FEC Viterbi.

        This handles both history and current block using the same logic:

            corrected pairing first,
            rotation second.
        """

        # History part
        if self.history_iq > 0:
            z_hist = self.make_symbols(0, self.history_iq, mode)
            z_hist = self.apply_rotation(z_hist, rotation)

            self.soft_float[: self.history_overlap : 2] = z_hist.real
            self.soft_float[1 : self.history_overlap : 2] = z_hist.imag

        # Current block part
        block_start = self.history_iq

        z_block = self.make_symbols(block_start, self.BLOCK_BITS, mode)
        z_block = self.apply_rotation(z_block, rotation)

        soft_start = self.history_overlap

        self.soft_float[soft_start::2] = z_block.real
        self.soft_float[soft_start + 1 :: 2] = z_block.imag

    def float_to_soft(self):
        """
        Convert float soft values in roughly [-1, +1] to unsigned soft bytes.

        0.0 maps to 128.
        Negative confidence maps below 128.
        Positive confidence maps above 128.
        """
        scaled = np.rint(self.soft_float * 127.0 + 128.0)
        scaled = np.clip(scaled, 0, 255)
        self.soft_u8[:] = scaled.astype(np.uint8)

    def decode_and_measure(self):
        self.float_to_soft()

        self.dec.generic_work(self.soft_caps, self.decoded_tmp_caps)
        self.enc.generic_work(self.decoded_tmp_caps, self.renc_caps)

        return self.compute_ber()

    def compute_ber(self):
        raw = self.soft_u8

        mask = raw != 128
        total = int(mask.sum())

        if total == 0:
            return 10.0

        hard = (raw > 127).astype(np.uint8)

        errors = int((hard[mask] != self.reencoded[mask]).sum())

        # Same scale factor as in your original code.
        return float(errors) / float(total) * 2.5

    def update_history(self):
        if self.history_iq <= 0:
            return

        # We consumed exactly BLOCK_BITS input samples.
        #
        # The lookahead sample was not consumed, so it must not become history yet.
        #
        # full_iq layout:
        #
        #   0 ... history_iq-1:
        #       previous history
        #
        #   history_iq ... history_iq+BLOCK_BITS-1:
        #       consumed current block
        #
        #   history_iq+BLOCK_BITS:
        #       lookahead, not consumed
        #
        consumed_end = self.history_iq + self.BLOCK_BITS

        self.prev_iq[:] = self.full_iq[
            consumed_end - self.history_iq : consumed_end
        ]

    def general_work(self, input_items, output_items):
        iq_in = input_items[0]
        bits_out = output_items[0]
        ber_out = output_items[1]

        # Need one extra sample for q_lookahead.
        if len(iq_in) < self.BLOCK_BITS + 1:
            return 0

        if len(bits_out) < self.BLOCK_BITS:
            return 0

        if len(ber_out) < self.BLOCK_BITS:
            return 0

        iq_window = iq_in[: self.BLOCK_BITS + 1]

        self.build_full_iq(iq_window)

        best_ber = 10.0
        best_name = "none"

        for name, mode, rotation in self.candidates:
            self.build_soft_input(mode, rotation)
            ber = self.decode_and_measure()

            if ber < best_ber:
                best_ber = ber
                best_name = name
                self.decoded_best[:] = self.decoded_tmp[-self.BLOCK_BITS :]

        bits_out[: self.BLOCK_BITS] = self.decoded_best
        ber_out[: self.BLOCK_BITS] = best_ber

        self.last_candidate = best_name
        self.last_ber = best_ber

        self.update_history()

        # Consume only BLOCK_BITS. The final input sample was only lookahead.
        self.consume(0, self.BLOCK_BITS)

        return self.BLOCK_BITS