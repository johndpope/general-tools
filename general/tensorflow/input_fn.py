from abc import ABCMeta, abstractmethod
import tensorflow as tf


class BaseInputFn():
    __metaclass__ = ABCMeta

    def __init__(self, labels_name='labels'):
        self.labels_name = labels_name

    # should return (features_names, dataset)
    @abstractmethod
    def _build_dataset(self):
        pass

    def __call__(self):
        features_names, dataset = self._build_dataset()

        iterator = dataset.make_one_shot_iterator()

        values = iterator.get_next()
        values_dict = dict(zip(features_names, values))

        labels = values_dict.pop(self.labels_name, None)
        return values_dict, labels


class StaticInputFn(BaseInputFn):
    def __init__(self, data_dict, batch_size=32, repeat=False, **args):
        super().__init__(**args)
        self.batch_size = batch_size
        self.data_dict = data_dict
        self.repeat = repeat

    def _build_dataset(self):
        keys = list(self.data_dict.keys())
        values = list(self.data_dict.values())

        dataset = tf.contrib.data.Dataset.from_tensor_slices(values)
        dataset = dataset.batch(values[0].shape[0])
        if self.repeat:
            dataset = dataset.repeat()

        return keys, dataset


class TFRecordsInputFn(BaseInputFn):
    def __init__(self, tfrecords_filenames, column_names, record_parser, batch=True, batch_size=64,
                 shuffle=True, shuffle_buffer_size=10000, repeat=True, **args):
        super().__init__(**args)
        self.tfrecords_filenames = tfrecords_filenames
        self.column_names = column_names
        self.record_parser = record_parser
        self.batch = batch
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        self.repeat = repeat

    def _build_dataset(self):
        dataset = tf.contrib.data.TFRecordDataset(self.tfrecords_filenames)

        dataset = dataset.map(self.record_parser)

        if self.shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)

        if self.batch:
            dataset = dataset.batch(self.batch_size)

        if self.repeat:
            dataset = dataset.repeat()

        return self.column_names, dataset


