import tensorflow as tf
import knime_extension as knext


# Function for building target-context pairs to train a skip-gram model (TF graph execution)

@tf.function
def skip_gram_obs_builder(encoded_corpus:tf.RaggedTensor, window_radius:tf.Tensor) -> tf.Tensor:
    
    couples = tf.TensorArray(tf.int64, size=tf.cast(tf.reduce_sum(encoded_corpus.row_lengths())*window_radius*2, tf.int32), 
                             element_shape=tf.TensorShape([2,]))
    m = tf.constant([0], dtype=tf.int32, shape=())
    i = tf.constant([0], dtype=tf.int64, shape=())
    zero_boundary = tf.constant([0], shape = (), dtype=tf.int64)
    for sequence in encoded_corpus:
        i-=i
        sequence_len = tf.shape(sequence, out_type=tf.int64)[0]
        for word in sequence:
            if word == 1:
                i+=1
                continue
            window = tf.range(start = tf.math.reduce_max([i-window_radius, zero_boundary]), 
                              limit = tf.math.reduce_min([i+window_radius+1, sequence_len]))
            for context_index in window:
                if context_index == i:
                    continue
                context = sequence[context_index]
                if context != 1:
                    couple = tf.stack([word, context], axis=0)
                    couples = couples.write(m, couple)
                    m+=1
            i+=1
    return couples.stack()[:m]



# Function for building contexts-target pairs to train a CBOW model (TF graph execution)
@tf.function 
def CBOW_obs_builder(encoded_corpus:tf.RaggedTensor, window_radius:tf.Tensor) -> tf.Tensor:
    
    final_array = tf.TensorArray(tf.int64, size=tf.cast(tf.reduce_sum(encoded_corpus.row_lengths()), tf.int32))
    m = tf.constant([0], dtype=tf.int32, shape=())
    i = tf.constant([0], dtype=tf.int64, shape=())
    z = tf.constant([0], dtype=tf.int32, shape=())
    filter_bool = tf.constant([True], tf.bool)
    for sequence in encoded_corpus:
        i-=i
        sequence_len = tf.shape(sequence, out_type=tf.int64)
        for word in sequence: 
            z-=z
            if word == 1: 
                i+=1
                continue
            window = tf.range(start = i - window_radius, 
                              limit = i + window_radius + 1)
            context_array = tf.TensorArray(tf.int64, size=tf.cast(window_radius*2, tf.int32), element_shape=tf.TensorShape([]))
            for context_index in window: 
                if context_index == i:
                    continue
                if context_index < 0 or context_index > sequence_len-1:
                    context = tf.constant(0, dtype=tf.int64)
                    context_array = context_array.write(z, context)
                    z+=1
                    continue
                context = sequence[context_index]
                context_array = context_array.write(z, context)
                z+=1
            context_set = context_array.stack()
            for integer in context_set:
                if integer <= 1:
                    pass
                else:
                    filter_bool = not filter_bool
                    break
            if filter_bool: 
                i+=1
                continue
            else:
                filter_bool = not filter_bool
            word = tf.expand_dims(word, 0)
            cbow_unit = tf.concat([context_set, word], 0)
            final_array = final_array.write(m, cbow_unit)
            context_array = context_array.close()
            i+=1
            m+=1
    return final_array.stack()[:m]
                
                    

# Model definition for skip-gram model
class Word2Vec_skipgram_keras(tf.keras.Model):
    def __init__(self, dict_size:int, embedding_size:int, hierarchical:bool, ns_sampl:None|int=None, *args, **kwargs) -> None: 
        super().__init__()
        self.input_word = tf.keras.layers.InputLayer(input_shape=(1,))
        self.input_embedding = tf.keras.layers.Embedding(input_dim=dict_size, output_dim=embedding_size, input_length=1, name = "EmbeddingsWord2Vec")
        
        if hierarchical: 
            self.output_tokens = tf.keras.layers.InputLayer(ragged=True, input_shape=(None,))
            self.output_embedding = tf.keras.layers.Embedding(input_dim=dict_size, output_dim=embedding_size)
        else:
            self.output_tokens = tf.keras.layers.InputLayer(input_shape = (ns_sampl+1))
            self.output_embedding = tf.keras.layers.Embedding(input_dim = dict_size, output_dim=embedding_size, input_length = ns_sampl+1)
            
        
        self.multiplicator_elwise = tf.keras.layers.Multiply()
        self.sigmoid = tf.keras.layers.Activation(tf.keras.activations.sigmoid)
        
    def __call__(self, inputs, *args, **kwargs):
        target = self.input_word(inputs[0])
        target = self.input_embedding(target)
        
        context = self.output_tokens(inputs[1])
        context = self.output_embedding(context)
        
        logits = self.multiplicator_elwise([target, context])
        logits = tf.math.reduce_sum(logits, axis = -1)
        
        probs = self.sigmoid(logits)
        return probs
            


# Model definition for CBOW model
class Word2Vec_CBOW_keras(tf.keras.Model):
    def __init__(self, dict_size:int, embedding_size:int, hierarchical:bool, window_size:int, ns_sampl:None|int=None, *args, **kwargs) -> None:
        super().__init__()
        self.input_context = tf.keras.layers.InputLayer(input_shape = (window_size*2,))
        self.input_embedding = tf.keras.layers.Embedding(input_dim=dict_size, output_dim=embedding_size, input_length=window_size*2, name="EmbeddingsWord2Vec")
        
        if hierarchical: 
            self.output_tokens = tf.keras.layers.InputLayer(ragged=True, input_shape = (None,))
            self.output_embedding = tf.keras.layers.Embedding(input_dim = dict_size, output_dim=embedding_size)
        else: 
            self.output_tokens = tf.keras.layers.InputLayer(input_shape = (ns_sampl+1))
            self.output_embedding = tf.keras.layers.Embedding(input_dim = dict_size, output_dim=embedding_size, input_length = ns_sampl+1)
        
        self.multiplicator_elwise = tf.keras.layers.Multiply()
        self.sigmoid = tf.keras.layers.Activation(tf.keras.activations.sigmoid)
    
    def __call__(self, inputs:list[tf.Tensor, tf.RaggedTensor], *args, **kwargs) -> tf.RaggedTensor:
        context = self.input_context(inputs[0])
        context = self.input_embedding(context)
        context = tf.math.reduce_mean(context, axis = 1)
        
        target = self.output_tokens(inputs[1])
        target = self.output_embedding(target)
        
        logits = self.multiplicator_elwise([context, target])
        logits = tf.math.reduce_sum(logits, axis = -1)
        
        probs = self.sigmoid(logits)
        return probs
    
    

# This callback Keras classed is passed to the fit method of the Keras model class in order to interact with KNIME during the training
# This allows the progress bar of the node to be progressively filled as epochs are finished and training approaches its end
class CallbackforKNIME(tf.keras.callbacks.Callback):
    def __init__(self, execution_context:knext.ExecutionContext, epoch_number:int, *args, **kwargs) -> None:
        super(CallbackforKNIME, self).__init__()
        self.exec_context = execution_context
        self.epoch_counter = 0
        self.epoch_number = epoch_number
        
    def on_train_begin(self, *args, **kwargs)-> None: 
        self.exec_context.set_progress(0.62, message="Training has begun")
        if self.exec_context.is_canceled():
            raise RuntimeError("Execution terminated by user")
    
    def on_epoch_end(self, *args, **kwargs)-> None:
        self.epoch_counter += 1
        if self.epoch_counter == 1:
            self.exec_context.set_progress(0.62+0.35*self.epoch_counter/self.epoch_number, message = f"{self.epoch_counter} epoch completed")
        else:
            self.exec_context.set_progress(0.62+0.35*self.epoch_counter/self.epoch_number, message = f"{self.epoch_counter} epochs completed")
    
    def on_train_batch_end(self, *args, **kwargs)-> None:
        if self.exec_context.is_canceled():
            raise RuntimeError("Execution terminated by user")

    
        
# Custom losses. Necessary since the BinaryCrossEntropy loss in Keras takes the mean of the element-wise cross-entropies computed along the rows 
# This leads to losing the information regarding the length of the path in the Huffman tree for the hierarchical softmax approach
# This is solved by performing an Hadamard product between the vector of the averages and the vector of the row lengths for the ragged tensor of the batch labels/logits

def custom_loss_word2vec(hier:bool):
    def hier_func(y_true, y_pred): 
        loss = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.NONE)(y_true, y_pred)
        loss = loss*tf.cast(y_true.row_lengths(), tf.float32)
        loss = tf.reduce_mean(loss, axis=0)
        return loss   
            
    
    def ns(y_true, y_pred):
        loss = tf.keras.losses.BinaryCrossentropy(reduction = tf.keras.losses.Reduction.NONE)(y_true, y_pred)
        loss = tf.reduce_mean(loss, axis=0)
        return loss

    if hier:
        return hier_func
    else:
        return ns
    
    

