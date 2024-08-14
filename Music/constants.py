DATASET_PATH = "deutschl/test"
SAVE_DIR = "dataset"
FILE_DATASET_COMBINED = "file_dataset_combined"
JSON_PATH = "mapping.json"
SEQUENCE_LENGTH = 64
# durations are expressed in quarter length
LIST_OF_ALLOWED_DURATION = [
    0.25, # 16th note
    0.5, # 8th note
    0.75,
    1.0, # quarter note
    1.5,
    2, # half note
    3,
    4 # whole note
]

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 5
BATCH_SIZE = 64
MODEL_PATH_LSTM = "/models/LSTM_model.h5"
MODEL_PATH_GRU = "/models/LSTM_model.h5"
MODEL_PATH_VANILLA_RNN = "/models/VanillaRNN_model.h5"

MIDI_PATH_LSTM_RNN='/midi/melody_model_lstm.mid'
MIDI_PATH_GRU_RNN='/midi/melody_model_gru.mid'
MIDI_PATH_VANILLA_RNN='/midi/melody_model_vanilla_rnn.mid'
