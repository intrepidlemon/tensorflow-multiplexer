import collections
import tensorflow as tf

Datasets = collections.namedtuple('Datasets', ['train', 'test'])

def get_decoder(bits=6):
    defaults = [ [0] for _ in range(bits) ] + [[0]]
    columns = [ i for i in range(bits) ] + ["label"]
    def decoder(line) :
        # Decode the line into its fields
        fields = tf.decode_csv(line, defaults, field_delim="\t")

        # Pack the result into a dictionary
        features = dict(zip(columns,fields))

        # Separate the label from the features
        label = features.pop('label')

        return tf.stack(columns[:-1]), label
    return decoder

def open_dataset(filename, bits=6):
    decoder = get_decoder(bits)
    ds = tf.data.TextLineDataset(filename).skip(1)
    return ds.map(decoder)

def create_train_test_datasets(train_file, test_file, bits=6):
    return Datasets(
        train=open_dataset(train_file, bits),
        test=open_dataset(test_file, bits),
    )

def load_multiplexer(data_dir, bits=6):
    return create_train_test_datasets(
        "{}/{}-bit/train.txt".format(data_dir, bits),
        "{}/{}-bit/test.txt".format(data_dir, bits),
        bits,
    )
