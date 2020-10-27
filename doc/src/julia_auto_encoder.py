import argparse

from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.layers import Input, GRU, RepeatVector, Activation, CuDNNGRU
from keras.layers import Dense, BatchNormalization, Embedding
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import os
import pandas as pd

def ae_models(maxlen, latent_dim, N, use_gpu=False):
    inputs = Input((maxlen,), name='Encoder_Inputs')
    encoded = Embedding(N, 
                        latent_dim, 
                        name='Char_Embedding', 
                        mask_zero=False)(inputs)
    encoded = BatchNormalization(name='BatchNorm_Encoder')(encoded)

    if use_gpu:
        _, state_h = CuDNNGRU(latent_dim, return_state=True)(encoded)
    else:
        _, state_h = GRU(latent_dim, return_state=True)(encoded)

    enc = Model(inputs=inputs, outputs=state_h, name='Encoder_Model')
    enc_out = enc(inputs)

    dec_inputs = Input(shape=(None,), name='Decoder_Inputs')
    decoded = Embedding(N, 
                        latent_dim, 
                        name='Decoder_Embedding', 
                        mask_zero=False)(dec_inputs)
    decoded = BatchNormalization(name='BatchNorm_Decoder_1')(decoded)

    if use_gpu:
        dec_out, _ = CuDNNGRU(latent_dim, 
                              return_state=True, 
                              return_sequences=True)(decoded, initial_state=enc_out)
    else:
        dec_out, _ = GRU(latent_dim, 
                         return_state=True, 
                         return_sequences=True)(decoded, initial_state=enc_out)

    dec_out = BatchNormalization(name='BatchNorm_Decoder_2')(dec_out)
    dec_out = Dense(N, activation='softmax', name='Final_Out')(dec_out)

    sequence_autoencoder = Model(inputs=[inputs, dec_inputs], outputs=dec_out)

    return sequence_autoencoder, enc

def build_and_train(args):
    seqs = get_seqs(args)
    N = len(np.unique(seqs))

    decoder_inputs = seqs[:,  :-1]
    Y = seqs[:, 1:  ]

    autoencoder, enc = ae_models(args.max_len, 
                                 args.dimension, 
                                 N, 
                                 use_gpu=args.use_gpu)

    autoencoder.compile(loss='sparse_categorical_crossentropy',
                        optimizer=Adam(lr=0.001, amsgrad=True),
                        metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_acc',
                              min_delta=0.0001,
                              patience=10,
                              verbose=1,
                              mode='auto',
                              restore_best_weights=True)

    autoencoder.fit([seqs, decoder_inputs],
                    np.expand_dims(Y, -1),
                    epochs = 100,
                    batch_size = 32,
                    validation_split=0.12,
                    callbacks=[early_stop],
                    shuffle=True)

    return autoencoder, enc

def chars_to_indices(data, tok=None, max_len=None):
    if max_len is None:
        max_len = max(data.apply(lambda x: len(x)))

    if tok is None:
        tok = Tokenizer(num_words=None, 
                        filters="", 
                        lower=False, 
                        split='', 
                        char_level=True)

    data = data.values
    tok.fit_on_texts(data)
    sequences = tok.texts_to_sequences(data)
    sequences = pad_sequences(sequences, 
                              maxlen=max_len, 
                              padding='post')
    sequences = np.array(sequences, dtype='int16')

    return sequences, tok

def get_seqs(args):
    with open(args.data, "r") as f: 
        funcs = f.read()

    funcs = process_code(funcs, args.maxlen)
    seqs, _ = chars_to_indices(funcs.code, max_len=args.maxlen)

    return seqs

def process_code(funcs, max_len=500):
    funcs = funcs.split(".jl\n")
    funcs = funcs[:-1] # remove trailing empty item
    funcs = pd.DataFrame([x.rsplit("\t",1) for x in funcs])
    funcs.columns = ['code','source']

    # limit length of code snippets, which are rarely huge
    funcs = funcs[funcs.code.str.len()<=max_len]
    funcs.reset_index(drop=True, inplace=True)

    funcs.source = funcs.source.apply(lambda x: x[x.index("julia/")+6:])
    funcs["top_folder"] = funcs.source.apply(lambda x: x[:x.index("/")])
    funcs['top2'] = funcs.source.apply(lambda x: '_'.join(x.split("/")[:2]))

    return funcs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto-encode Julia source code.')
    parser.add_argument('-D', '--dir', default=os.getcwd(),
                        help='Directory for trained models.')
    parser.add_argument('-d', '--data', required=True,
                        help='The extracted source code snippets for encoding.')
    parser.add_argument('--maxlen', default=500, type=int,
                        help='Maximum code snippet length.')
    parser.add_argument('--dimension', default=64, type=int,
                        help='Encoding dimension for representation.')
    parser.add_argument('--use_gpu', action="store_true",
                        help='Should we use the GPU if available?')
    parser.add_argument('-m', '--mode', default='train', options=['train','encode']
                        help='Mode for auto-encoder [train, encode].')

    args = parser.parse_args()

    if args.mode == 'train':
        autoenc, enc = build_and_train(args)

        autoencoder.save(os.path.join(args.dir,"autoencoder.h5"))
        enc.save(os.path.join(args.dir,"encoder.h5"))
    elif args.mode == 'encode':
        if not os.path.isabs(args.data):
            args.data = os.path.join(os.getcwd(), args.data)

        data_dir = os.path.dirname(args.data)
        enc = load_model(os.path.join(args.dir, "encoder.h5"))
        seqs = get_seqs(args)
        encoded_reps = enc.predict(seqs)
        encoded_reps = pd.DataFrame(encoded_reps)
        encoded_reps.to_csv(os.path.join(data_dir, "encoded_reps.csv"), index=False)

