from gnuradio import gr
import pmt


class CcsdsImageAssembler(gr.basic_block):
    def __init__(self, width=1568):
        gr.basic_block.__init__(
            self,
            name="ccsds_image_assembler",
            in_sig=None,
            out_sig=None,
        )

        self.width = int(width)

        self._byte_buffer = bytearray()   # incoming raw chunks
        self._image_buf = bytearray()     # only full image rows

        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.handle_msg)

    def handle_msg(self, msg):
        if pmt.is_pair(msg) is False:
            return

        data = pmt.cdr(msg)
        if pmt.is_u8vector(data) is False:
            return

        payload = bytes(pmt.u8vector_elements(data))
        self._byte_buffer.extend(payload)

        # extract full rows based on width
        full = (len(self._byte_buffer) // self.width) * self.width
        if full <= 0:
            return
        
        print(".", end='', flush=True)

        self._image_buf.extend(self._byte_buffer[:full])
        del self._byte_buffer[:full]

    def get_bytes(self):
        return bytes(self._image_buf)

    def get_dimensions(self):
        height = len(self._image_buf) // self.width
        return self.width, height

    def clear(self):
        self._byte_buffer.clear()
        self._image_buf.clear()

    def size(self):
        return len(self._image_buf)