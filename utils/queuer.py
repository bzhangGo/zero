# coding: utf-8

"""
The Queue function mainly deals with reading and preparing dataset in a multi-processing manner.
We didnot use the built-in tensorflow function Dataset because it lacks of flexibility.
The function defined below is mainly inspired by https://github.com/ixlan/machine-learning-data-pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from multiprocessing import Process, Queue

TERMINATION_TOKEN = "<DONE>"


def create_iter_from_queue(queue, term_token):

    while True:
        input_data_chunk = queue.get()
        if input_data_chunk == term_token:
            # put it back to the queue to let other processes that feed
            # from the same one to know that they should also break
            queue.put(term_token)
            break
        else:
            yield input_data_chunk


def combine_reader_to_processor(reader, preprocessor):
    for data_chunk in reader:
        yield preprocessor(data_chunk)


class EnQueuer(object):
    def __init__(self,
                 reader,
                 preprocessor,
                 worker_processes_num=1,
                 input_queue_size=5,
                 output_queue_size=5
                 ):
        if worker_processes_num < 0:
            raise ValueError("worker_processes_num must be a "
                             "non-negative integer.")

        self.worker_processes_number = worker_processes_num
        self.preprocessor = preprocessor
        self.input_queue_size = input_queue_size
        self.output_queue_size = output_queue_size
        self.reader = reader

    # make the queue iterable
    def __iter__(self):
        return self._create_processed_data_chunks_gen(self.reader)

    def _create_processed_data_chunks_gen(self, reader_gen):
        if self.worker_processes_number == 0:
            itr = self._create_single_process_gen(reader_gen)
        else:
            itr = self._create_multi_process_gen(reader_gen)
        return itr

    def _create_single_process_gen(self, data_producer):
        return combine_reader_to_processor(data_producer, self.preprocessor)

    def _create_multi_process_gen(self, reader_gen):
        term_tokens_received = 0
        output_queue = Queue(self.output_queue_size)
        workers = []

        if self.worker_processes_number > 1:
            term_tokens_expected = self.worker_processes_number - 1
            input_queue = Queue(self.input_queue_size)
            reader_worker = _ParallelWorker(reader_gen, input_queue)
            workers.append(reader_worker)

            # adding workers that will process the data
            for _ in range(self.worker_processes_number - 1):
                # since data-chunks will appear in the queue, making an iterable
                # object over it
                queue_iter = create_iter_from_queue(input_queue,
                                                    TERMINATION_TOKEN)

                data_itr = combine_reader_to_processor(queue_iter, self.preprocessor)
                proc_worker = _ParallelWorker(data_chunk_iter=data_itr,
                                              queue=output_queue)
                workers.append(proc_worker)
        else:
            term_tokens_expected = 1

            data_itr = combine_reader_to_processor(reader_gen, self.preprocessor)
            proc_worker = _ParallelWorker(data_chunk_iter=data_itr,
                                          queue=output_queue)
            workers.append(proc_worker)

        for pr in workers:
            pr.daemon = True
            pr.start()

        while True:
            data_chunk = output_queue.get()
            if data_chunk == TERMINATION_TOKEN:
                term_tokens_received += 1
                # need to received all tokens in order to be sure that
                # all data has been processed
                if term_tokens_received == term_tokens_expected:
                    for pr in workers:
                        pr.join()
                    break
                continue
            yield data_chunk


class _ParallelWorker(Process):
    """Worker to execute data reading or processing on a separate process."""

    def __init__(self, data_chunk_iter, queue):
        super(_ParallelWorker, self).__init__()
        self._data_chunk_iterable = data_chunk_iter
        self._queue = queue

    def run(self):
        for data_chunk in self._data_chunk_iterable:
            self._queue.put(data_chunk)
        self._queue.put(TERMINATION_TOKEN)
