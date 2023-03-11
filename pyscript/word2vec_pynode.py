import logging
import sys
import os

sys.path.append(os.path.realpath(__file__))
import knime.extension as knext
import pandas as pd

LOGGER = logging.getLogger(__name__)
import sklearn.feature_extraction as sk_fe
import tensorflow as tf
import numpy as np
import hierarchical_softmax
from utility_tf_word2vec import (
    skip_gram_obs_builder,
    CBOW_obs_builder,
    Word2Vec_skipgram_keras,
    Word2Vec_CBOW_keras,
    custom_loss_word2vec,
    CallbackforKNIME,
)

python_category = knext.category(
    path="/community",
    level_id="word2vec",
    name="Word2Vec",
    description="",
    icon="word2vec.png",
)


@knext.parameter_group(label="Algorithm Hyperparameters")
class algorithm_param:
    embedding_size = knext.IntParameter(
        label="Embedding size",
        default_value=20,
        min_value=1,
        description="Change the embedding size of the two Word2Vec embedding layers \
                                                                                                            (for target and context words, respectively) in order to get speed (smaller number) or \
                                                                                                            performance (larger number).",
    )

    window_size = knext.IntParameter(
        label="Window size (radius)",
        min_value=1,
        default_value=2,
        description='Choose the radius of the window size that represents how far from the target \
                                                                                                            word Word2Vec looks. The context window always has the target word at the center, \
                                                                                                            and the number that can be set determines the "radius" \
                                                                                                            of the window, meaning that the actual number of context words considered is twice what is inserted.',
    )

    negative_sample_int = knext.IntParameter(
        label="Number of negative samples",
        default_value=5,
        min_value=1,
        description="The negative sampling approach is a way to simplify the computational complexity \
                                                                                                                             of vanilla Word2Vec while trying to introduce noise in the models in order to regularize it. \
                                                                                                                             You can choose the number of negative samples.",
    )

    hierarchical_soft = knext.BoolParameter(
        label="Hierarchical Softmax",
        default_value=False,
        description="Activate hierarchical softmax in place of negative sampling. This option thus deactivates negative sampling.",
    )

    algorithm_selection = knext.StringParameter(
        label="Word2Vec algorithm selection",
        description="Choose between CBOW (target as output) and skip-gram (context as output) Word2Vec implementation.",
        enum=["CBOW", "skip-gram"],
        default_value="skip-gram",
    )


@knext.parameter_group(label="Dictionary Parameters")
class dictionary_param:
    word_survival_bool = knext.BoolParameter(
        label="Word Survival Function",
        default_value=True,
        description="Whether to use a word survival function \
                                                                                                            to reduce the size of the vocabulary by prioritizing rarer words.",
    )

    word_survival_rate = knext.DoubleParameter(
        label="Sampling rate for Word Survival Function (if flagged)",
        max_value=0.1,
        min_value=10 ** -10,
        default_value=10 ** -4,
        description="Set the sampling rate for the Word Survival function, the higher it \
                                                         gets the more words are included in the dictionary. Default value is 10^-3. Max value is 0.1.",
    )

    minimum_freq = knext.IntParameter(
        label="Minimum Frequency",
        min_value=0,
        default_value=5,
        description="Minimum corpus frequency below which a word in the dictionary is not considered. \
                                                Set it to 0 if filtering according to minimum frequency is not needed.",
    )


@knext.parameter_group(label="Training parameters")
class training_param:
    nr_epoch = knext.IntParameter(
        label="Epochs",
        min_value=1,
        default_value=10,
        description="Number of epochs for model training. The more epoch, the longer time to train, linearly.",
    )
    batch_size = knext.IntParameter(
        label="Batch size",
        min_value=8,
        default_value=32,
        description="The batch size you want to set to train the Word2Vec model.",
    )
    adam_lr = knext.DoubleParameter(
        label="Adam learning rate",
        description="Set the learning rate for the Adam optimizer. The actual step in the parameter space is dynamic during training.",
        min_value=10 ** -10,
        default_value=10 ** -4,
    )


@knext.parameter_group(label="Word2Vec parameters")
class Word2VecParam:
    group_1 = algorithm_param()
    group_2 = dictionary_param()


@knext.node(
    name="Word2Vec Learner (Tensorflow)",
    node_type=knext.NodeType.LEARNER,
    icon_path="word2vec.png",
    category=python_category,
)
@knext.input_table(
    name="Table with string column",
    description="A KNIME table with a string column to use for Word2Vec training",
)
@knext.output_table(
    name="Table of embeddings",
    description="A KNIME table with three columns: the index of the token/the word, the token itself and the embedding for the token as a collection (KNIME native list).",
)
class Word2VecLearner:
    """
    This Python-based node extracts embeddings from a fitted Word2Vec model, giving the possibility of choosing between the two Word2Vec algorithms, CBOW and skip-gram.
    To perform the actual training, hierarchical softmax and negative sampling are both available.
    The node uses Tensorflow as engine to speed up the pre-processing and to fit the model.
    Given the presence of a CUDA compatible NVIDIA GPU, training can be performed on the GPU.
    """

    def is_string(
            column,
    ):  # Filter columns visible in the column_param. Only string ones are visible
        return column.ktype == knext.string()

    input_column = knext.ColumnParameter(
        label="Column selection (String type)",
        description="Select which document type column you want to use to train the model.",
        column_filter=is_string,
    )

    word2vec_conf = Word2VecParam()
    training_conf = training_param()

    set_seed = knext.BoolParameter(
        label="Set seed",
        description="Set seeds for the whole node.",
        default_value=True,
    )
    seed = knext.IntParameter(
        label="Seed",
        description="Choose the seed number, if you do not want the default one.",
        default_value=1234,
    )

    # Search for devices which are visible by Tensorflow, lets the user choose which to use. Defaults to first one, which is CPU
    devices_options = [
        ":".join(x.name.split(":")[-2:]) for x in tf.config.list_physical_devices()
    ]
    default_device = devices_options[0]
    tf_device = knext.StringParameter(
        label="Device for Tensorflow model fit",
        description="Choose the device where to run the fit for the Word2Vec model; only the visible devices are available. \
                                                                                            Notice that the indexes next to the device name are just identifiers for the device itself.",
        enum=devices_options,
        default_value=default_device,
    )

    def configure(self, configuration_context, input_table_1):

        if knext.string() not in [x.ktype for x in list(input_table_1)]:
            raise knext.InvalidParametersError(
                "The input table does not have any string columns. You need to have a string column for this node."
            )

        # raise warning if no column is selected for input and first string one is considered
        if self.input_column is None:
            configuration_context.set_warning(
                "Autoguess, using the first string column of the input table"
            )
        elif (
                input_table_1[self.input_column].ktype != knext.string()
        ):  # Users may change the data type of a column retaining the same name, without this check things would break in that case
            raise knext.InvalidParametersError(
                "The column you previously selected is no longer of string type"
            )

        # What happens if node settings have a device not available on local machine?
        if self.tf_device not in self.devices_options:
            configuration_context.set_warning(
                "The device previously chosen is not available on the local machine, reverting to execution on default device"
            )
            self.tf_device = self.default_device

        return knext.Schema(
            ktypes=[knext.string(), knext.ListType(knext.double())],
            names=["Token", "Embedding"],
        )

    def execute(self, exec_context, input_1):

        # Set seed if needed
        if self.set_seed:
            # Sets all random seeds for the program (Python, NumPy legacy and TensorFlow)
            tf.keras.utils.set_random_seed(self.seed)
            # Initialize the new Numpy generator and set a seed if needed
            random_gen = np.random.default_rng(self.seed)
        else:
            random_gen = np.random.default_rng()

        # If input column not specified, use the first one which is string type
        if self.input_column is None:
            self.input_column = [
                x.name for x in list(input_1.schema) if x.ktype == knext.string()
            ][0]

        # Reading the input and converting it to a Pandas Dataframe, then getting a Series of strings by using the key from the ColumParameter selection
        input_1 = input_1.to_pandas()
        corpus = input_1[self.input_column].tolist()

        # Use sci-kit learn to build document term matrix, extract also the string for each token.
        count_vectorizer_obj = sk_fe.text.CountVectorizer(
            lowercase=False, token_pattern=r"(?u)\b\w[-\w]+\b"
        ).fit(corpus)
        document_tm_mtrx = count_vectorizer_obj.transform(corpus).toarray()
        token_strings = count_vectorizer_obj.get_feature_names()

        # Build term frequency table
        tf_series = document_tm_mtrx.sum(axis=0)
        tf_table = pd.DataFrame(zip(token_strings, tf_series), columns=["Token", "TF"])

        # Sample the dictionary words according to word survival function
        # The token column of the resulting table is the dictionary we are passing to TextVectorization Layer
        tf_table["RF"] = tf_table["TF"] / tf_table["TF"].sum()
        tf_table["keep_probability"] = (
                                               (
                                                       (tf_table["RF"] / self.word2vec_conf.group_2.word_survival_rate)
                                                       ** (1 / 2)
                                               )
                                               + 1
                                       ) * (self.word2vec_conf.group_2.word_survival_rate / tf_table["RF"])
        tf_table["keep_probability"] = [
            1 if elem > 1 else elem for elem in tf_table["keep_probability"]
        ]
        tf_table["label"] = random_gen.binomial(
            1, p=tf_table["keep_probability"]
        )  # A binomial with n=1 is equal to a Bernoulli
        tf_table = tf_table[tf_table["label"] == 1]
        tf_table.drop(labels=["keep_probability", "label"], axis=1, inplace=True)

        # We implement the hard coded threshold for the word frequency
        tf_table = tf_table[tf_table["TF"] >= self.word2vec_conf.group_2.minimum_freq]

        exec_context.set_progress(
            0.10, message="Sampling and filtering on the vocabulary completed"
        )
        if exec_context.is_canceled():
            raise RuntimeError("Execution terminated by user")

        # We turn the text data into integers that are then digestable by the Keras Embedding Layer
        with tf.device(self.default_device):
            vectorizer = tf.keras.layers.TextVectorization(
                vocabulary=tf_table["Token"], standardize=None, ragged=True
            )
            integer_corpus = vectorizer(corpus)
            exec_context.set_progress(0.15)
            if exec_context.is_canceled():
                raise RuntimeError("Execution terminated by user")

        # Add a column to the table for the integer token of each word
        tf_table["int_token"] = range(2, vectorizer.vocabulary_size())

        # This part uses Tensorflow graph execution to dramatically speed up composition of target/context pairs.
        # The functions used come from the utility_tf_word2vec module

        with tf.device(self.default_device):
            # Accumulator variables, parameter weights etc. (everything which keeps some state) as
            # Tensorflow variables
            m = tf.Variable(0, dtype=tf.int32, shape=())
            i = tf.Variable(0, dtype=tf.int64, shape=())
            if self.word2vec_conf.group_1.algorithm_selection == "skip-gram":
                target_context_skipgr = skip_gram_obs_builder(
                    integer_corpus,
                    tf.constant(self.word2vec_conf.group_1.window_size, dtype=tf.int64),
                    m,
                    i
                ).numpy()
                exec_context.set_progress(
                    0.40,
                    message="Scan of the corpus to build skip-gram pairs completed",
                )
            if self.word2vec_conf.group_1.algorithm_selection == "CBOW":
                # z and filter_bool are additional variables needed for the control flow of the function
                z = tf.Variable(0, dtype=tf.int32, shape=())
                filter_bool = tf.Variable([True], dtype=tf.bool)
                context_target_CBOW = CBOW_obs_builder(
                    integer_corpus,
                    tf.constant(self.word2vec_conf.group_1.window_size, dtype=tf.int64),
                    m,
                    i,
                    z,
                    filter_bool
                ).numpy()
                exec_context.set_progress(
                    0.40, message="Scan of the corpus to build CBOW pairs completed"
                )

        if exec_context.is_canceled():
            raise RuntimeError("Execution terminated by user")

        # If negative sampling, we calculate the probability of a token being sampled as negative class
        if not self.word2vec_conf.group_1.hierarchical_soft:
            tf_table["sampling_prob"] = (tf_table["TF"] / tf_table["TF"].sum()) ** (
                    3 / 4
            )
            tf_table["sampling_prob"] = tf_table["sampling_prob"] / (
                tf_table["sampling_prob"].sum()
            )

            # Sampling with np random choice and weights
            negative_sampl = list()
            sampling_vector = np.array(range(2, len(tf_table) + 2))
            for _ in range(len(target_context_skipgr)):
                negative_sampl.append(
                    random_gen.choice(
                        sampling_vector,
                        size=self.word2vec_conf.group_1.negative_sample_int,
                        replace=False,
                        p=tf_table["sampling_prob"],
                        shuffle=False,
                    )
                )
            negative_sampl = np.array(negative_sampl)

            # Create labels. For negative sampling the first is 1, the others are 0s
            labels = [1] + [0] * self.word2vec_conf.group_1.negative_sample_int
            labels = np.repeat(
                np.expand_dims(labels, axis=0), repeats=len(negative_sampl), axis=0
            )

            if self.word2vec_conf.group_1.algorithm_selection == "skip-gram":
                training_input = tf.constant(
                    target_context_skipgr[:, 0], dtype=tf.int64
                )
                training_output = tf.constant(
                    np.concatenate(
                        (
                            np.expand_dims(target_context_skipgr[:, -1], axis=1),
                            negative_sampl,
                        ),
                        axis=1,
                    ),
                    dtype=tf.int64,
                )
            else:
                training_input = tf.constant(
                    context_target_CBOW[:, 0:-1], dtype=tf.int64
                )
                training_output = tf.constant(
                    np.concatenate(
                        (
                            np.expand_dims(context_target_CBOW[:, -1], axis=1),
                            negative_sampl,
                        ),
                        axis=1,
                    ),
                    dtype=tf.int64,
                )
            training_label = tf.constant(labels, dtype=tf.int64)

            exec_context.set_progress(
                0.60, message="Negative sampling completed. Train is about to start..."
            )

        # If hierarchical softmax, we call the factory method of the Tree class from the hierarchical softmax module
        # This allows us to build the Huffman tree data structure we exploit for the hierarchical softmax
        if self.word2vec_conf.group_1.hierarchical_soft:
            frequency_tuples = [
                (token, freq)
                for token, freq in tf_table[["int_token", "TF"]].itertuples(
                    index=False, name=None
                )
            ]
            binary_tree = hierarchical_softmax.Tree.huffman_builder(frequency_tuples)

            # Two slightly different branches depending on whether the data comes from skip-gram pre-processing or from CBOW pre-processing
            if self.word2vec_conf.group_1.algorithm_selection == "skip-gram":
                list_dataset = target_context_skipgr.tolist()
            else:
                list_dataset = context_target_CBOW.tolist()

            # For every output word, the path_finder recursive method from the Tree class is called,
            # and a list of tuples (internal token and right direction in the path) is returned
            # The elements of this list of tuples are extracted and used to build the (ragged) tensors needed to train the Word2Vec model with a hierarchical softmax
            labels_hr = list()
            tokens_hr = list()
            input_hr = list()
            for row in list_dataset:
                label_path = list()
                token_path = list()
                path_to_cont = binary_tree.path_finder(row[-1])
                for label, internal_token in path_to_cont:
                    label_path.append(label)
                    token_path.append(-internal_token)
                labels_hr.append(label_path)
                tokens_hr.append(token_path)
                if self.word2vec_conf.group_1.algorithm_selection == "skip-gram":
                    input_hr.append(row[0])
                else:
                    input_hr.append(row[0:-1])
            training_input = tf.constant(input_hr)
            training_output = tf.ragged.constant(tokens_hr, row_splits_dtype=tf.int32)
            training_label = tf.ragged.constant(labels_hr, row_splits_dtype=tf.int32)
            exec_context.set_progress(
                0.6,
                "Hierarchical Softmax data structure built, path to leaves extracted. Train is about to start...",
            )

        if exec_context.is_canceled():
            raise RuntimeError("Execution terminated by user")

        # Depending on the algorithm selection (Skip-gram or CBOW) two different model classes are used for training. The classes are loaded from the utility_tf_word2vec module
        if self.word2vec_conf.group_1.algorithm_selection == "skip-gram":
            model = Word2Vec_skipgram_keras(
                vectorizer.vocabulary_size(),
                self.word2vec_conf.group_1.embedding_size,
                self.word2vec_conf.group_1.hierarchical_soft,
                self.word2vec_conf.group_1.negative_sample_int,
            )
        else:
            model = Word2Vec_CBOW_keras(
                vectorizer.vocabulary_size(),
                self.word2vec_conf.group_1.embedding_size,
                self.word2vec_conf.group_1.hierarchical_soft,
                self.word2vec_conf.group_1.window_size,
                self.word2vec_conf.group_1.negative_sample_int,
            )

        # Notice the callbacks argument in the fit, which is used to interact with KNIME during training (specifically, the progress bar of the node)
        # A callback keras class in the utility_tf_word2vec module is defined for that
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.training_conf.adam_lr),
            loss=custom_loss_word2vec(self.word2vec_conf.group_1.hierarchical_soft),
            metrics=[],
        )
        with tf.device(self.tf_device):
            model.fit(
                [training_input, training_output],
                training_label,
                batch_size=self.training_conf.batch_size,
                epochs=self.training_conf.nr_epoch,
                verbose=1,
                callbacks=CallbackforKNIME(exec_context, self.training_conf.nr_epoch),
            )

        embeddings = model.get_layer("EmbeddingsWord2Vec").get_weights()[0][2:]
        embeddings = embeddings.tolist()
        embeddings_table = pd.DataFrame(
            {"Token": tf_table["Token"].tolist(), "Embedding": embeddings},
            tf_table["int_token"],
            dtype=object,
        )

        return knext.Table.from_pandas(embeddings_table)  # row id bool to add here as soon as extension is updated
