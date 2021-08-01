# coding: utf-8
# author: Biao Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import tensorflow as tf

from logging import Handler

try:
  unicode
  _unicode = True
except NameError:
  _unicode = False


class TFBoardWriter(object):
  """TensorBoard Writer Class, to help visualization"""

  def __init__(self, sess, output_path):
    self._sess = sess
    self._writer = tf.summary.FileWriter(output_path)

  def scalar(self, name, value, step):
    """Note, only scalar summary is supported"""

    summ = tf.Summary(value=[
      tf.Summary.Value(tag=name, simple_value=value),
    ])

    self._writer.add_summary(
      summ,
      step
    )

  def close(self):
    self._writer.flush()
    self._writer.close()


class TFStreamHandler(Handler):
  """
  A handler class which writes logging records, appropriately formatted,
  to a stream. Note that this class does not close the stream, as
  sys.stdout or sys.stderr may be used.

  Adding control to the delay of flush, you can't flush to the google cloud
  storage too frequently. Network is a big bottleneck!
  """

  def __init__(self, stream=None, delay=1000):
    """
    Initialize the handler.

    If stream is not specified, sys.stderr is used.
    """
    Handler.__init__(self)
    if stream is None:
      stream = sys.stderr
    self.stream = stream
    self._delay = delay
    self._emit_counter = 0

  def flush(self):
    """
    Flushes the stream.
    """
    self.acquire()
    try:
      if self.stream and hasattr(self.stream, "flush"):
        self.stream.flush()
    finally:
      self.release()

  def emit(self, record):
    """
    Emit a record.

    If a formatter is specified, it is used to format the record.
    The record is then written to the stream with a trailing newline.  If
    exception information is present, it is formatted using
    traceback.print_exception and appended to the stream.  If the stream
    has an 'encoding' attribute, it is used to determine how to do the
    output to the stream.
    """
    try:
      msg = self.format(record)
      stream = self.stream
      fs = "%s\n"
      if not _unicode:  # if no unicode support...
        stream.write(fs % msg)
      else:
        try:
          if (isinstance(msg, unicode) and
            getattr(stream, 'encoding', None)):
            ufs = u'%s\n'
            try:
              stream.write(ufs % msg)
            except UnicodeEncodeError:
              # Printing to terminals sometimes fails. For example,
              # with an encoding of 'cp1251', the above write will
              # work if written to a stream opened or wrapped by
              # the codecs module, but fail when writing to a
              # terminal even when the codepage is set to cp1251.
              # An extra encoding step seems to be needed.
              stream.write((ufs % msg).encode(stream.encoding))
          else:
            stream.write(fs % msg)
        except UnicodeError:
          stream.write(fs % msg.encode("UTF-8"))

      # add delayed emitting
      self._emit_counter += 1
      if self._emit_counter % self._delay == 0:
        self.flush()

    except (KeyboardInterrupt, SystemExit):
      raise
    except:
      self.handleError(record)
