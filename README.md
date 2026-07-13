# GNU flowgraphs for Meteor M-N2.x LRPT

The project is using blocks from gr-satellites, follow the installation here.

https://gr-satellites.readthedocs.io/en/latest/installation_intro.html

I wanted to learn about OQPSK demodulation and picked Meteor M-N2.x LRPT as
target. This series of the Meteor weather satellites has two members as of 2026.
Meteor M N2-3 and Meteor M N2-4 active on 137.9MHz with a backup frequency of
137.1MHz.

A detailed tutorial about reception is available at
https://www.a-centauri.com/articoli/meteor-satellite-reception, this repository
is only about the extraction of data.

The project is organized into smaller GNU radio hier blocks (files with .grc and
a matching .py and block.yml) and OOT modules (files without .grc).

# Usage from command line

You will need a cf32 input file. These can be created using
[Satdump](https://github.com/SatDump/SatDump)'s Recording features. The file
name has the format of

```
<datetime>_<sps>SPS_<frequency>.cf32
```

For example

```
2026-02-13_07-39-09_256000SPS_137900000Hz.cf32
```

then use:

```
python3 meteor/meteor_extractor.py 2026-02-13_07-39-09_256000SPS_137900000Hz.cf32
```

This should generate an image file in the same directory.

# Usage from GNU radio

![overview](pics/overview.jpg)

Open gnuradio-companion with

```bash
GRC_HIER_PATH=./meteor GRC_BLOCKS_PATH=. gnuradio-companion
```

The main entry point for demonstration is meteor_demo.grc.

Select the input file in the flowgraph, and set the sample rate parameter to
your sps (e.g. 256000). Then click the play button.

![output](pics/output.jpg)

If things go well you should see something like this.

The satellite has 6 image channels, the LRPT downlink uses 3 of these. As of
this writing channels 1, 2 and 4 are active. You can connect a CCSDS Image
decoder to these pins.

I was greatly inspired by how Satdump implements the protocol, and tried
reimplement its metrics as well, these are exposed as 'doppler hz' from the
costas loop, 'SNR (db)' from soft symbols and 'BER' from the viterbi stage.

It's worth checking out the OQPSK demodulator pipeline as well, which is a 1-1
reimplementation of what Satdump does, only in GNU radio terms. I tried to trim
it down to only this version of Meteor, since the original one supports the
previous generation as well.

My implementation doesn't work with 80k samples, only the 72k line is
implemented.

The later stages follow the descriptions found in
http://jmfriedt.free.fr/glmf_meteor_eng.pdf.

I also used https://github.com/Digitelektro/MeteorDemod for inspiration. I think
the jpg decoder part is ported from there.

I might want to get rid of the gr-satellites dependency later, but it comes with
a proper reed-solomon decoder as a wrapper around
[libfec](https://github.com/daniestevez/gr-satellites/tree/1358c09ee1924b2c407ddc60859ea083add925d8/lib/libfec).
It would be hard to port this to python, but we will see.

