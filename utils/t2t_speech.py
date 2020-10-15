# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import h5py
import argparse
import functools
import numpy as np
import scipy.signal
import tensorflow as tf
import tensorflow.contrib as tc

import yaml
import librosa
from pathlib import Path
from scipy.io import wavfile
from subprocess import call


"""
The preprocessing procedure given in tensor2tensor, maybe this one works better
"""


def add_delta_deltas(filterbanks, name=None):
    """Compute time first and second-order derivative channels.
      Args:
        filterbanks: float32 tensor with shape [batch_size, len, num_bins, 1]
        name: scope name
      Returns:
        float32 tensor with shape [batch_size, len, num_bins, 3]
    """
    delta_filter = np.array([2, 1, 0, -1, -2])
    delta_delta_filter = scipy.signal.convolve(delta_filter, delta_filter, "full")

    delta_filter_stack = np.array(
        [[0] * 4 + [1] + [0] * 4, [0] * 2 + list(delta_filter) + [0] * 2,
         list(delta_delta_filter)],
        dtype=np.float32).T[:, None, None, :]

    delta_filter_stack /= np.sqrt(
        np.sum(delta_filter_stack ** 2, axis=0, keepdims=True))

    filterbanks = tf.nn.conv2d(
        filterbanks, delta_filter_stack, [1, 1, 1, 1], "SAME", data_format="NHWC",
        name=name)
    return filterbanks


def compute_mel_filterbank_features(
        waveforms,
        sample_rate=16000, dither=1.0 / np.iinfo(np.int16).max, preemphasis=0.97,
        frame_length=25, frame_step=10, fft_length=None,
        window_fn=functools.partial(tf.signal.hann_window, periodic=True),
        lower_edge_hertz=80.0, upper_edge_hertz=7600.0, num_mel_bins=80,
        log_noise_floor=1e-3, apply_mask=True):
    """Implement mel-filterbank extraction using tf ops.
      Args:
        waveforms: float32 tensor with shape [batch_size, max_len]
        sample_rate: sampling rate of the waveform
        dither: stddev of Gaussian noise added to waveform to prevent quantization
          artefacts
        preemphasis: waveform high-pass filtering constant
        frame_length: frame length in ms
        frame_step: frame_Step in ms
        fft_length: number of fft bins
        window_fn: windowing function
        lower_edge_hertz: lowest frequency of the filterbank
        upper_edge_hertz: highest frequency of the filterbank
        num_mel_bins: filterbank size
        log_noise_floor: clip small values to prevent numeric overflow in log
        apply_mask: When working on a batch of samples, set padding frames to zero
      Returns:
        filterbanks: a float32 tensor with shape [batch_size, len, num_bins, 1]
    """
    # `stfts` is a complex64 Tensor representing the short-time Fourier
    # Transform of each signal in `signals`. Its shape is
    # [batch_size, ?, fft_unique_bins]
    # where fft_unique_bins = fft_length // 2 + 1

    # Find the wave length: the largest index for which the value is !=0
    # note that waveforms samples that are exactly 0.0 are quite common, so
    # simply doing sum(waveforms != 0, axis=-1) will not work correctly.
    wav_lens = tf.reduce_max(
        tf.expand_dims(tf.range(tf.shape(waveforms)[1]), 0) *
        tf.to_int32(tf.not_equal(waveforms, 0.0)),
        axis=-1) + 1
    if dither > 0:
        waveforms += tf.random_normal(tf.shape(waveforms), stddev=dither)
    if preemphasis > 0:
        waveforms = waveforms[:, 1:] - preemphasis * waveforms[:, :-1]
        wav_lens -= 1
    frame_length = int(frame_length * sample_rate / 1e3)
    frame_step = int(frame_step * sample_rate / 1e3)
    if fft_length is None:
        fft_length = int(2 ** (np.ceil(np.log2(frame_length))))

    stfts = tf.signal.stft(
        waveforms,
        frame_length=frame_length,
        frame_step=frame_step,
        fft_length=fft_length,
        window_fn=window_fn,
        pad_end=True)

    stft_lens = (wav_lens + (frame_step - 1)) // frame_step
    masks = tf.to_float(tf.less_equal(
        tf.expand_dims(tf.range(tf.shape(stfts)[1]), 0),
        tf.expand_dims(stft_lens, 1)))

    # An energy spectrogram is the magnitude of the complex-valued STFT.
    # A float32 Tensor of shape [batch_size, ?, 257].
    magnitude_spectrograms = tf.abs(stfts)

    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    linear_to_mel_weight_matrix = (
        tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
            upper_edge_hertz))
    mel_spectrograms = tf.tensordot(
        magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    # Note: Shape inference for tensordot does not currently handle this case.
    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(
        linear_to_mel_weight_matrix.shape[-1:]))

    log_mel_sgram = tf.log(tf.maximum(log_noise_floor, mel_spectrograms))

    if apply_mask:
        log_mel_sgram *= tf.expand_dims(tf.to_float(masks), -1)

    return tf.expand_dims(log_mel_sgram, -1, name="mel_sgrams")


def audio_encode(s):
    """Transform a string with a filename into a list of float32.
    Args:
      s: path to the file with a waveform.
    Returns:
      samples: list of int16s
    """

    def convert_to_wav(in_path, out_path, extra_args=None):
        if not os.path.exists(out_path):
            # TODO(dliebling) On Linux, check if libsox-fmt-mp3 is installed.
            args = ["sox", "--rate", "16k", "--bits", "16", "--channel", "1"]
            if extra_args:
                args += extra_args
            call(args + [in_path, out_path])

    # Make sure that the data is a single channel, 16bit, 16kHz wave.
    # TODO(chorowski): the directory may not be writable, this should fallback
    # to a temp path, and provide instructions for installing sox.
    if s.endswith(".mp3"):
        out_filepath = s[:-4] + ".wav"
        convert_to_wav(s, out_filepath, ["--guard"])
        s = out_filepath
    elif not s.endswith(".wav"):
        out_filepath = s + ".wav"
        convert_to_wav(s, out_filepath)
        s = out_filepath

    rate, data = wavfile.read(s)
    assert len(data.shape) == 1
    if data.dtype not in [np.float32, np.float64]:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data.astype(np.float32)


def audio_encode_must_c(wav_path, offset, duration, sample_rate=16000):
    """
    Encoding audio files into float list given the offset and duration
    We assume the sample rate to be 16k, this is consistant with MUST-C dataset

    Note, must_c is already in wav format
    """
    # load data, sr=None enforce to use the native sample rate
    data, rate = librosa.load(wav_path, sr=None, offset=offset, duration=duration)
    assert len(data.shape) == 1 and rate == sample_rate

    if data.dtype not in [np.float32, np.float64]:
        data = data.astype(np.float32) / np.iinfo(data.dtype).max
    return data.astype(np.float32)


def get_hparams(model_hparams):
    def add_if_absent(p, attr, value):
        if not hasattr(p, attr):
            p.add_hparam(attr, value)

    p = model_hparams
    # Filterbank extraction in bottom instead of preprocess_example is faster.
    add_if_absent(p, "audio_preproc_in_bottom", False)
    # The trainer seems to reserve memory for all members of the input dict
    add_if_absent(p, "audio_keep_example_waveforms", False)
    add_if_absent(p, "audio_sample_rate", 16000)
    add_if_absent(p, "audio_preemphasis", 0.97)
    add_if_absent(p, "audio_dither", 1.0 / np.iinfo(np.int16).max)
    add_if_absent(p, "audio_frame_length", 25.0)
    add_if_absent(p, "audio_frame_step", 10.0)
    add_if_absent(p, "audio_lower_edge_hertz", 20.0)
    add_if_absent(p, "audio_upper_edge_hertz", 8000.0)
    add_if_absent(p, "audio_num_mel_bins", 80)
    add_if_absent(p, "audio_add_delta_deltas", True)
    add_if_absent(p, "num_zeropad_frames", 250)

    return p


def shape_list(x):
    # Copied from Tensor2Tensor
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def preprocess_example(example, hparams):
    p = hparams
    if p.audio_preproc_in_bottom:
        example["inputs"] = tf.expand_dims(
            tf.expand_dims(example["waveforms"], -1), -1)
    else:
        waveforms = tf.expand_dims(example["waveforms"], 0)
        mel_fbanks = compute_mel_filterbank_features(
            waveforms,
            sample_rate=p.audio_sample_rate,
            dither=p.audio_dither,
            preemphasis=p.audio_preemphasis,
            frame_length=p.audio_frame_length,
            frame_step=p.audio_frame_step,
            lower_edge_hertz=p.audio_lower_edge_hertz,
            upper_edge_hertz=p.audio_upper_edge_hertz,
            num_mel_bins=p.audio_num_mel_bins,
            apply_mask=False)
        if p.audio_add_delta_deltas:
            mel_fbanks = add_delta_deltas(mel_fbanks)
        fbank_size = shape_list(mel_fbanks)
        assert fbank_size[0] == 1

        # This replaces CMVN estimation on data
        var_epsilon = 1e-09
        mean = tf.reduce_mean(mel_fbanks, keepdims=True, axis=1)
        variance = tf.reduce_mean(tf.squared_difference(mel_fbanks, mean),
                                  keepdims=True, axis=1)
        mel_fbanks = (mel_fbanks - mean) * tf.rsqrt(variance + var_epsilon)

        # Later models like to flatten the two spatial dims. Instead, we add a
        # unit spatial dim and flatten the frequencies and channels.
        example["inputs"] = tf.concat([
            tf.reshape(mel_fbanks, [fbank_size[1], fbank_size[2], fbank_size[3]]),
            tf.zeros((p.num_zeropad_frames, fbank_size[2], fbank_size[3]))], 0)

    if not p.audio_keep_example_waveforms:
        del example["waveforms"]
    return example


def text_process(text, char=True):
    """
    Pre-processing the given input text, if char is true, then we use character processing
    Or, keep space segmentation, just as in MT settings
    :param text: the original text input
    :param char: whether use character processing, (default True)
    :return:
    """
    text = text.strip()

    if not char:
        return text.split()
    else:
        splitted = list(text)
        splitted = [ch if ch != ' ' else '<SPACE>' for ch in splitted]
        return splitted


def build_model(hparams, bs=64):
    """Compile model for speech feature extraction"""
    batch_outputs = []
    placeholders = []
    for pidx in range(bs):
        waveform = tf.placeholder(tf.float32, [None], "waveform_{}".format(pidx))
        placeholders.append(waveform)

        example = {
            "waveforms": waveform,
        }
        # example["inputs"] of shape [1, len, num_bins, 3]
        example = preprocess_example(example, hparams)
        # flattern inputs
        ishp = shape_list(example["inputs"])
        example["inputs"] = tf.reshape(example["inputs"], [ishp[0], -1])

        batch_outputs.append(example["inputs"])
    return batch_outputs, placeholders


def process_data(data_path, save_path, hparams, char=True, bs=64):
    """
    Reading the audio, text and save them into pkl format
    This is for LibriSpeech, might need change for other datasets
    """
    dir_path = Path(data_path)
    txt_list = [f for f in dir_path.glob('**/*.txt') if f.is_file()]

    print('Number of audio txt paths:', len(txt_list))

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    sess = tf.Session(config=config)

    audios = []
    txts = []

    hf = h5py.File(save_path+".audio.h5", 'w')
    txt_writer = open(save_path + ".txt", 'w')
    ref_writer = open(save_path + ".ref.txt", 'w')

    h = 0
    bs_iter = 0
    batch_feed_dict = {}

    # define the placeholders and compiles modeling graph
    batch_outputs, placeholders = build_model(hparams, bs)

    for i, txt in enumerate(txt_list):
        print('Text#:', i + 1)
        with open(str(txt)) as reader:
            txt_path, txt_filename = os.path.split(str(txt))
            for line in reader:
                # line format: 84-121123-0013 ASKED MORREL YES
                txt_tokens = line.strip().split()
                audio_filename = os.path.join(txt_path, txt_tokens[0] + ".flac")

                batch_feed_dict[placeholders[bs_iter]] = audio_encode(audio_filename)
                bs_iter += 1

                target = text_process(' '.join(txt_tokens[1:]), char=char)
                txts.append(len(target))
                txt_writer.write(' '.join(target) + "\n")
                ref_writer.write((''.join(target)).replace('<SPACE>', ' ') + "\n")

                if bs_iter >= bs:
                    sources = sess.run(batch_outputs[:bs_iter], feed_dict=batch_feed_dict)

                    for source in sources:
                        audios.append(len(source))

                        hf.create_dataset("audio_{}".format(h), data=source)
                        h += 1

                    bs_iter = 0
                    batch_feed_dict = {}

    if bs_iter > 0:
        sources = sess.run(batch_outputs[:bs_iter], feed_dict=batch_feed_dict)

        for source in sources:
            audios.append(len(source))

            hf.create_dataset("audio_{}".format(h), data=source)
            h += 1

        bs_iter = 0
        batch_feed_dict = {}

    print('Avg number audio length:', len(audios), np.mean(audios))
    print('Avg number text length:', len(txts), np.mean(txts))

    hf.close()
    txt_writer.close()
    ref_writer.close()


def process_data_must_c(data_path, save_path, hparams, char=True, bs=64):
    """
    Reading the audio, text and save them into pkl format
    This is for Must-C corpus, might need change for other datasets

    data_path: e.g., train: h5, txt, wav
    """
    # loading yaml alignment-file
    dir_path = Path(data_path)
    yaml_list = [f for f in dir_path.glob('txt/*.yaml') if f.is_file]
    assert len(yaml_list) == 1, 'must be 1 and only 1 yaml file'
    yaml_path = yaml_list[0]

    print("Starting loading YAML alignment file")
    with open(str(yaml_path), 'r') as yaml_reader:
        sample_list = yaml.safe_load(yaml_reader)
    print("Finishing Loading.")

    print('Number of audio txt paths:', len(sample_list))

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    sess = tf.Session(config=config)

    audios = []

    hf = h5py.File(save_path+".audio.h5", 'w')

    h = 0
    bs_iter = 0
    batch_feed_dict = {}

    # define the placeholders and compiles modeling graph
    batch_outputs, placeholders = build_model(hparams, bs)

    for i, sample in enumerate(sample_list):
        print('Sample#:', i + 1)

        batch_feed_dict[placeholders[bs_iter]] = audio_encode_must_c(
            data_path + "/wav/" + sample['wav'],
            sample['offset'],
            sample['duration'],
            sample_rate=16000)
        bs_iter += 1

        if bs_iter >= bs:
            sources = sess.run(batch_outputs[:bs_iter], feed_dict=batch_feed_dict)

            for source in sources:
                audios.append(len(source))

                hf.create_dataset("audio_{}".format(h), data=source)
                h += 1

            bs_iter = 0
            batch_feed_dict = {}

    if bs_iter > 0:
        sources = sess.run(batch_outputs[:bs_iter], feed_dict=batch_feed_dict)

        for source in sources:
            audios.append(len(source))

            hf.create_dataset("audio_{}".format(h), data=source)
            h += 1

        bs_iter = 0
        batch_feed_dict = {}

    print('Avg number audio length:', len(audios), np.mean(audios))

    hf.close()


def process_data_librit(data_path, save_path, hparams, char=True, bs=64):
    """
    Reading the audio, text and save them into pkl format
    This is for LibiriSpeech En-Fr translation corpus, might need change for other datasets

    data_path: e.g., train: h5, txt, wav
    """
    # loading yaml alignment-file
    dir_path = Path(data_path)
    align_list = [f for f in dir_path.glob('*.meta') if f.is_file]
    assert len(align_list) == 1, 'must be 1 and only 1 alignment file'
    align_path = align_list[0]

    print("Starting loading alignment file")
    with open(str(align_path), 'r') as align_reader:
        sample_list = []
        header_line = True
        for sample in align_reader:
            if header_line:
                header_line = False
                continue
            sample_list.append(sample.strip().split())
    print("Finishing Loading.")

    print('Number of audio txt paths:', len(sample_list))

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    sess = tf.Session(config=config)

    audios = []

    hf = h5py.File(save_path+".audio.h5", 'w')

    h = 0
    bs_iter = 0
    batch_feed_dict = {}

    # define the placeholders and compiles modeling graph
    batch_outputs, placeholders = build_model(hparams, bs)

    for i, sample in enumerate(sample_list):
        print('Sample#:', i + 1)

        batch_feed_dict[placeholders[bs_iter]] = audio_encode(data_path + "/audiofiles/" + sample[-2] + ".wav")
        bs_iter += 1

        if bs_iter >= bs:
            sources = sess.run(batch_outputs[:bs_iter], feed_dict=batch_feed_dict)

            for source in sources:
                audios.append(len(source))

                hf.create_dataset("audio_{}".format(h), data=source)
                h += 1

            bs_iter = 0
            batch_feed_dict = {}

    if bs_iter > 0:
        sources = sess.run(batch_outputs[:bs_iter], feed_dict=batch_feed_dict)

        for source in sources:
            audios.append(len(source))

            hf.create_dataset("audio_{}".format(h), data=source)
            h += 1

        bs_iter = 0
        batch_feed_dict = {}

    print('Avg number audio length:', len(audios), np.mean(audios))

    hf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Speech Dataset Preprocessing')
    parser.add_argument('--word', action="store_true", help='enable word-based text processing')
    parser.add_argument('input', type=str, help='the input file path')
    parser.add_argument('output', type=str, help='the output file name')

    parser.add_argument('--melbins', type=int, default=80, help='speech feature dimension')
    parser.add_argument('--numpad', type=int, default=250, help='feature padding to audios')
    parser.add_argument('--bs', type=int, default=256, help='batch size')
    parser.add_argument('--noise', action="store_true", help='enable random noised features')
    parser.add_argument('--dataset', type=str, help='processing speech translation dataset: must_c, librit, libiris')

    # it turns out that the speech preprocessing is very  important to seq2seq models
    # sensitivity is the biggest problem.
    # with good features, the model can learn translation very quickly; but with a bad one,
    #    no useful translation will be produced, or very unstable gradients will occur.
    # By default, we use 40 melbins, 0 numpaddings, and no noise.
    # The data are saved by library h5py, one after another in the same input order.

    args = parser.parse_args()

    hparams = get_hparams(tc.training.HParams())

    hparams.audio_num_mel_bins = args.melbins
    hparams.num_zeropad_frames = args.numpad
    if not args.noise:
        hparams.audio_dither = 0.0

    print("# loading data start")
    if args.dataset == "must_c":
        process_data_must_c(
            args.input,
            args.output,
            hparams,
            char=(not args.word),
            bs=args.bs,
        )
    elif args.dataset == "librit":
        process_data_librit(
            args.input,
            args.output,
            hparams,
            char=(not args.word),
            bs=args.bs,
        )
    elif args.dataset == "libris":
        process_data(
            args.input,
            args.output,
            hparams,
            char=(not args.word),
            bs=args.bs,
        )
    else:
        raise Exception("Unkown dataset {}".format(args.dataset))
    print("# data has been saved into {}".format(args.output))
