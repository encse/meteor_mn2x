from gnuradio import gr
import pmt
from dataclasses import dataclass
from typing import List, Optional

from decode_jpeg import decode_14_blocks


# ---------------- CONSTANTS ----------------

BLOCKS_PER_LINE = 14
BLOCK_WIDTH = 8 * 14          # 112
BLOCK_HEIGHT = 8
IMAGE_WIDTH = BLOCKS_PER_LINE * BLOCK_WIDTH  # 1568


def fresh_line():
    return [[0] * IMAGE_WIDTH for _ in range(BLOCK_HEIGHT)]

@dataclass
class Segment:
    MCUN: int
    QF: int
    payload: bytes


def parse_segment(data: bytes) -> Segment:
    if len(data) < 14:
        raise ValueError("Segment too short")

    MCUN = data[8]
    QF = data[13]

    return Segment(
        MCUN=MCUN,
        QF=QF,
        payload=data[14:]
    )


# ---------------- GNU RADIO BLOCK ----------------

class CcsdsImageDecoder(gr.basic_block):
    """
    Input:  message port "in"  (u8vector CCSDS space packet)
    Output: message port "out" (u8vector, one image row per message)
    """

    def __init__(self):
        gr.basic_block.__init__(self, name="ccsds_image_decoder", in_sig=[], out_sig=[])

        self.current_line: Optional[List[List[int]]] = None
        self.packet_id = None
        self.offset = 0

        self.message_port_register_in(pmt.intern("in"))
        self.message_port_register_out(pmt.intern("out"))
        self.set_msg_handler(pmt.intern("in"), self._handle_msg)

        self.packet_sequence_count_key = pmt.intern("space_packet.packet_sequence_count")
        

    def _handle_msg(self, msg):
        meta = pmt.car(msg)
        data = pmt.cdr(msg) 
        payload = bytes(pmt.u8vector_elements(data))
        
        sequence_count = pmt.dict_ref(meta, self.packet_sequence_count_key, pmt.PMT_NIL)
        sequence_count = pmt.to_long(sequence_count)

      
        self._process_packet(sequence_count, payload)

    def _emit_rows(self, rows: List[List[int]]):
        for row in rows:
            vec = pmt.init_u8vector(len(row), row)
            out_msg = pmt.cons(pmt.PMT_NIL, vec)
            self.message_port_pub(pmt.intern("out"), out_msg)

    def _process_packet(self, sequence_count, payload: bytes):
        seg = parse_segment(payload)

        pixels_8x112 = decode_14_blocks(seg.payload, seg.QF)

        packet_idx_in_line = seg.MCUN // 14
        x0 = packet_idx_in_line * BLOCK_WIDTH

        # compute an ever increasing packet_id
        if self.packet_id is None:
            self.packet_id = sequence_count
        else:
            while self.packet_id >= sequence_count + self.offset:
                self.offset += 16384

        new_packet_id = sequence_count + self.offset

        # emit missing lines
        diff = new_packet_id - self.packet_id
        missing_lines = diff // 43
        if missing_lines > 0:
            if self.current_line is not None:
                self._emit_rows(self.current_line)
                self.current_line = None 

            for _ in range(missing_lines):
                    self._emit_rows(fresh_line())

        self.packet_id = new_packet_id

        # New line but previous incomplete -> flush
        if packet_idx_in_line == 0 and self.current_line is not None:
            self._emit_rows(self.current_line)
            self.current_line = None


        if self.current_line is None:
            self.current_line = fresh_line()

        for r in range(BLOCK_HEIGHT):
            self.current_line[r][x0:x0 + BLOCK_WIDTH] = pixels_8x112[r]

        # End of line
        if packet_idx_in_line == (BLOCKS_PER_LINE - 1):
            self._emit_rows(self.current_line)
            self.current_line = None

