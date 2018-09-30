from typing import Dict

import tensorflow as tf


def _prepare(serialized_example):
    pass


class TFRDataset(object):
    """
    Represents a whole dataset consisting of one or many TFR-files.
    """

    def __init__(self, files_with_sizes: Dict[str, int], mode: str):
        self._mode = mode
        self._files = []
        self._size = 0

        for file, size in files_with_sizes.items():
            self._files.append(file)
            self._size += size

        assert self._files, f'No files for {mode} dataset'
        assert self._size, f'Empty {mode} dataset'

    @property
    def mode(self):
        return self._mode

    @property
    def size(self):
        return self._size

    def process(self,
                is_training: bool,
                process_record_fn,
                batch_size: int,
                shuffle_buffer_size: int,
                num_cpu_cores: int,
                num_epochs: int,
                num_parallel_batches: int):
        """
        Returns deserialized and preprocessed TF dataset.
        :param is_training: Whether the input is for training.
        :param process_record_fn: A function that takes deserialized features,
            preprocesses them, and then returns the corresponding
            (preprocessed_input, preprocessed_target) pair(s).
        :param batch_size: The number of examples per batch.
        :param shuffle_buffer_size: The buffer size to use when shuffling records.
        :param num_cpu_cores: Number of CPU cores.
        :param num_epochs: The number of epochs to repeat the dataset.
        :param num_parallel_batches: Number of batches preparing in parallel.
        :return: Dataset of (input_image, target_image) pairs ready for iteration.
        """
        files_dataset = tf.data.Dataset\
            .from_tensor_slices(self._files)\
            .shuffle(len(self._files))

        def tfrecord_dataset(filename):
            return tf.data.TFRecordDataset(filename, buffer_size=0)

        dataset = files_dataset.apply(
            tf.contrib.data.parallel_interleave(
                tfrecord_dataset,
                cycle_length=num_cpu_cores,
                sloppy=True))

        if is_training:
            dataset = dataset\
                .shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)\
                .repeat(num_epochs)
        else:
            dataset = dataset.repeat(num_epochs)

        def process_record(serialized_example):
            return process_record_fn(
                _prepare(serialized_example))

        dataset = dataset.apply(
            tf.contrib.data.map_and_batch(
                process_record,
                batch_size,
                num_parallel_batches=num_parallel_batches))

        return dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
