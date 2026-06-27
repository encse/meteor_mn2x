#!/usr/bin/env python3

import argparse
import math
import os
import re
import sys

import numpy as np
import pmt
from PIL import Image, ImageOps
from gnuradio import blocks
from gnuradio import gr

import meteor_lrpt
import ccsds_image_decoder

from ccsds_image_assembler import CcsdsImageAssembler


FILENAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})_(?P<time>\d{2}-\d{2}-\d{2})_(?P<sps>\d+)SPS_(?P<freq>\d+)Hz\.cf32$"
)

CHANNEL_PORTS = {
    1: "msu_mr_1",
    2: "msu_mr_2",
    3: "msu_mr_3",
    4: "msu_mr_4",
}

METEOR_SWATH_KM = 2800.0
EARTH_RADIUS_KM = 6371.0

SATELLITES = {
    "meteor-m2-3": {
        "altitude_km": 828.0,
        "swath_km": METEOR_SWATH_KM,
    },
    "meteor-m2-4": {
        "altitude_km": 819.0,
        "swath_km": METEOR_SWATH_KM,
    },
}

DEFAULT_SATELLITE = "meteor-m2-3"


def parse_input_filename(input_path):
    base_name = os.path.basename(input_path)
    match = FILENAME_RE.match(base_name)

    if match is None:
        raise ValueError(
            "Input filename must match "
            "YYYY-MM-DD_HH-MM-SS_<sample_rate>SPS_<frequency>Hz.cf32"
        )

    return {
        "date": match.group("date"),
        "time": match.group("time"),
        "sample_rate": int(match.group("sps")),
        "frequency_hz": int(match.group("freq")),
    }


def build_output_directory(input_path):
    meta = parse_input_filename(input_path)
    directory = os.path.dirname(input_path)
    output_name = f"{meta['date']}_{meta['time']}_meteor_lrpt"

    if directory == "":
        return output_name

    return os.path.join(directory, output_name)


def build_channel_output_filename(output_dir, channel_index, corrected=False):
    suffix = "_corrected" if corrected else ""
    return os.path.join(output_dir, f"channel_{channel_index}{suffix}.png")


def build_composite_output_filename(output_dir, composite_name, corrected=False):
    suffix = "_corrected" if corrected else ""
    return os.path.join(output_dir, f"composite_{composite_name}{suffix}.png")


def gamma_from_alpha(alpha, earth_radius_km, satellite_radius_km):
    return math.atan2(
        earth_radius_km * math.sin(alpha),
        satellite_radius_km - earth_radius_km * math.cos(alpha),
    )


def correct_cross_track_geometry(
    img,
    altitude_km,
    swath_km,
):
    """
    Horizontally corrects a Meteor LRPT cross-track image.

    Assumptions:
    - input image columns are approximately uniform in satellite scan/view angle
    - corrected output columns are uniform in ground distance
    - corrected output width is equal to swath width in pixels

    For Meteor:
    - 2800 km swath -> 2800 px corrected image width
    """

    if altitude_km <= 0:
        raise ValueError("altitude_km must be positive")

    if swath_km <= 0:
        raise ValueError("swath_km must be positive")

    width, height = img.size

    if width < 2:
        raise ValueError("input image width must be at least 2 pixels")

    output_width = int(round(swath_km))

    if output_width <= 0:
        raise ValueError("output_width must be positive")

    earth_radius = EARTH_RADIUS_KM
    satellite_radius = earth_radius + float(altitude_km)

    alpha_max = (float(swath_km) / 2.0) / earth_radius
    alpha_horizon = math.acos(earth_radius / satellite_radius)

    if alpha_max >= alpha_horizon:
        raise ValueError(
            f"swath_km is too large for altitude_km. "
            f"alpha_max={math.degrees(alpha_max):.2f} deg, "
            f"horizon={math.degrees(alpha_horizon):.2f} deg"
        )

    gamma_max = gamma_from_alpha(
        alpha_max,
        earth_radius,
        satellite_radius,
    )

    src = np.asarray(img)

    if src.ndim != 2:
        raise ValueError("This function expects a grayscale image")

    # Corrected output x axis: uniform ground distance across the full swath.
    alpha_values = np.linspace(-alpha_max, alpha_max, output_width)

    # Convert each ground position back to satellite scan angle, because the raw
    # image columns are assumed to be uniform in scan angle.
    gamma_values = np.array(
        [
            gamma_from_alpha(abs(alpha), earth_radius, satellite_radius)
            * (1.0 if alpha >= 0 else -1.0)
            for alpha in alpha_values
        ],
        dtype=np.float64,
    )

    # Map gamma back to source x coordinate in the raw image.
    x_src = (gamma_values / gamma_max + 1.0) * 0.5 * (width - 1)

    x0 = np.floor(x_src).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    x0 = np.clip(x0, 0, width - 1)

    t = x_src - x0

    left = src[:, x0]
    right = src[:, x1]

    corrected = (left * (1.0 - t) + right * t).astype(np.uint8)

    return Image.fromarray(corrected, mode="L")

def equalize_composite(img):
    if img.mode != "RGB":
        raise ValueError("equalize_composite expects an RGB image")

    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()

    y = ImageOps.equalize(y)

    return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")

def crop_to_common_size(images):
    min_width = min(img.size[0] for img in images)
    min_height = min(img.size[1] for img in images)

    cropped = []

    for img in images:
        if img.size != (min_width, min_height):
            img = img.crop((0, 0, min_width, min_height))
        cropped.append(img)

    return cropped


def save_composites(output_dir, channel_images, corrected=False, equalize=False):
    composite_specs = {
        "221": (2, 2, 1),
        "321": (3, 2, 1),
    }

    for composite_name, channel_order in composite_specs.items():
        missing_channels = [
            channel_index
            for channel_index in set(channel_order)
            if channel_index not in channel_images
        ]

        if missing_channels:
            suffix = "corrected " if corrected else ""
            print(
                f"Skipping {suffix}composite {composite_name}: "
                f"missing channels {missing_channels}",
                file=sys.stderr,
            )
            continue

        red = channel_images[channel_order[0]]
        green = channel_images[channel_order[1]]
        blue = channel_images[channel_order[2]]

        red, green, blue = crop_to_common_size([red, green, blue])

        composite = Image.merge("RGB", (red, green, blue))

        if equalize:
            composite = equalize_composite(composite)

        output_file = build_composite_output_filename(
            output_dir,
            composite_name,
            corrected=corrected,
        )

        composite.save(output_file)

        label = f"composite {composite_name}"
        if corrected:
            label += " corrected"

        print(f"Saved {label}: {output_file}")


class MessageNullSink(gr.basic_block):
    def __init__(self):
        gr.basic_block.__init__(
            self,
            name="message_null_sink",
            in_sig=None,
            out_sig=None,
        )
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.handle_msg)

    def handle_msg(self, msg):
        pass


class MeteorAllChannelExtractor(gr.top_block):
    def __init__(self, input_path, sample_rate):
        gr.top_block.__init__(self, "meteor_all_channel_extractor")

        self.input_path = input_path
        self.sample_rate = int(sample_rate)

        self.blocks_file_source_0 = blocks.file_source(
            gr.sizeof_gr_complex,
            self.input_path,
            False,
        )

        self.meteor_lrpt_0 = meteor_lrpt.meteor_lrpt(
            sample_rate=self.sample_rate
        )

        self.null_sink_constellation_0 = blocks.null_sink(gr.sizeof_gr_complex)
        self.null_sink_ber_0 = blocks.null_sink(gr.sizeof_float)
        self.null_sink_frequency_0 = blocks.null_sink(gr.sizeof_float)
        self.null_sink_snr_0 = blocks.null_sink(gr.sizeof_float)

        self.connect((self.blocks_file_source_0, 0), (self.meteor_lrpt_0, 0))

        self.connect((self.meteor_lrpt_0, 0), (self.null_sink_constellation_0, 0))
        self.connect((self.meteor_lrpt_0, 1), (self.null_sink_ber_0, 0))
        self.connect((self.meteor_lrpt_0, 2), (self.null_sink_frequency_0, 0))
        self.connect((self.meteor_lrpt_0, 3), (self.null_sink_snr_0, 0))

        self.image_decoders = {}
        self.image_assemblers = {}

        for channel_index, channel_port in CHANNEL_PORTS.items():
            decoder = ccsds_image_decoder.CcsdsImageDecoder()
            assembler = CcsdsImageAssembler(width=1568)

            self.image_decoders[channel_index] = decoder
            self.image_assemblers[channel_index] = assembler

            self.msg_connect(
                (self.meteor_lrpt_0, channel_port),
                (decoder, "in"),
            )

            self.msg_connect(
                (decoder, "out"),
                (assembler, "in"),
            )


def main():
    parser = argparse.ArgumentParser(
        description="Decode Meteor M-N2 LRPT image channels from a .cf32 IQ recording."
    )

    parser.add_argument(
        "input_file",
        help="Input file in the format YYYY-MM-DD_HH-MM-SS_<sample_rate>SPS_<frequency>Hz.cf32",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Default: <date>_<time>_meteor_lrpt next to input file.",
    )

    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Rotate all output images by 180 degrees.",
    )
    
    parser.add_argument(
        "--equalize",
        action="store_true",
        help="Apply equalization to composite images.",
    )


    parser.add_argument(
        "--sat",
        choices=sorted(SATELLITES.keys()),
        default=DEFAULT_SATELLITE,
        help=(
            "Satellite used for geometry correction. "
            f"Default: {DEFAULT_SATELLITE}."
        ),
    )

    args = parser.parse_args()

    try:
        meta = parse_input_filename(args.input_file)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    output_dir = args.output_dir

    if output_dir is None:
        output_dir = build_output_directory(args.input_file)

    os.makedirs(output_dir, exist_ok=True)

    sat = SATELLITES[args.sat]

    tb = MeteorAllChannelExtractor(
        input_path=args.input_file,
        sample_rate=meta["sample_rate"],
    )

    print(f"Input file: {args.input_file}")
    print(f"Sample rate: {meta['sample_rate']}")
    print(f"Frequency: {meta['frequency_hz']} Hz")
    print(f"Satellite: {args.sat}")
    print(f"Altitude: {sat['altitude_km']} km")
    print(f"Swath: {sat['swath_km']} km")
    print(f"Corrected width: {int(round(sat['swath_km']))} px")
    print(f"Output directory: {output_dir}")
    print(f"Rotate: {args.rotate}")

    tb.run()

    saved_any = False
    raw_channel_images = {}
    corrected_channel_images = {}

    for channel_index in sorted(CHANNEL_PORTS.keys()):
        assembler = tb.image_assemblers[channel_index]

        data = assembler.get_bytes()
        width, height = assembler.get_dimensions()

        if height <= 0:
            print(
                f"Channel {channel_index}: no complete image rows were assembled.",
                file=sys.stderr,
            )
            continue

        img = Image.frombytes("L", (width, height), data)

        if args.rotate:
            img = img.transpose(Image.Transpose.ROTATE_180)


        raw_output_file = build_channel_output_filename(
            output_dir,
            channel_index,
            corrected=False,
        )

        img.save(raw_output_file)
        raw_channel_images[channel_index] = img
        saved_any = True

        print(f"Channel {channel_index}: saved {raw_output_file}")

        try:
            corrected_img = correct_cross_track_geometry(
                img,
                altitude_km=sat["altitude_km"],
                swath_km=sat["swath_km"],
            )
        except ValueError as exc:
            print(
                f"Channel {channel_index}: correction failed: {exc}",
                file=sys.stderr,
            )
            continue

        corrected_output_file = build_channel_output_filename(
            output_dir,
            channel_index,
            corrected=True,
        )

        corrected_img.save(corrected_output_file)
        corrected_channel_images[channel_index] = corrected_img

        print(f"Channel {channel_index}: saved {corrected_output_file}")

    if raw_channel_images:
        save_composites(
            output_dir,
            raw_channel_images,
            corrected=False,
            equalize=args.equalize,
        )

    if corrected_channel_images:
        save_composites(
            output_dir,
            corrected_channel_images,
            corrected=True,
            equalize=args.equalize,
        )

    if not saved_any:
        print("No complete images were assembled.", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())