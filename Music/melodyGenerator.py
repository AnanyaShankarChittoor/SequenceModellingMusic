import numpy as np
import json
import tensorflow.keras as keras
import music21 as m21
from constants import MODEL_PATH_LSTM, MODEL_PATH_GRU, MODEL_PATH_VANILLA_RNN, JSON_PATH, SEQUENCE_LENGTH

class MusicGenerator:

    def __init__(self, model_path, mapping_path=JSON_PATH):
        self.model = keras.models.load_model(model_path)

        with open(mapping_path, "r") as file:
            self.note_mapping = json.load(file)

        self.start_tokens = ["/"] * SEQUENCE_LENGTH

    def compose_melody(self, initial_notes, steps, max_seq_len, temp):
        initial_notes = initial_notes.split()
        melody = initial_notes
        current_sequence = self.start_tokens + initial_notes

        current_sequence = [self.note_mapping[note] for note in current_sequence]

        for _ in range(steps):
            current_sequence = current_sequence[-max_seq_len:]
            input_sequence = keras.utils.to_categorical(current_sequence, num_classes=len(self.note_mapping))
            input_sequence = input_sequence[np.newaxis, ...]

            predicted_probs = self.model.predict(input_sequence)[0]
            next_note_int = self._pick_note_with_temperature(predicted_probs, temp)

            current_sequence.append(next_note_int)
            next_note = [note for note, index in self.note_mapping.items() if index == next_note_int][0]

            if next_note == "/":
                break

            melody.append(next_note)

        return melody

    def _pick_note_with_temperature(self, probabilities, temperature):
        scaled_probs = np.log(probabilities) / temperature
        scaled_probs = np.exp(scaled_probs) / np.sum(np.exp(scaled_probs))
        return np.random.choice(len(probabilities), p=scaled_probs)

    def save_melody(self, melody, duration_per_step=0.25, file_format="midi", filename="melody.mid"):
        stream = m21.stream.Stream()
        current_note = None
        duration_counter = 1

        for i, note in enumerate(melody):
            if note != "_" or i + 1 == len(melody):
                if current_note is not None:
                    total_duration = duration_per_step * duration_counter
                    if current_note == "r":
                        music_element = m21.note.Rest(quarterLength=total_duration)
                    else:
                        music_element = m21.note.Note(int(current_note), quarterLength=total_duration)
                    stream.append(music_element)
                    duration_counter = 1
                current_note = note
            else:
                duration_counter += 1

        stream.write(file_format, filename)

if __name__ == "__main__":
    lstm_generator = MusicGenerator(model_path='LSTM_model.h5')
    gru_generator = MusicGenerator(model_path='GRU_model.h5')
    rnn_generator = MusicGenerator(model_path='VanillaRNN_model.h5')

    seed = "60 _ 60 _ 67 _ 67 _ 69 _ 69 _ 67 _ _ " #Twinkle Twinkle
    num_steps = 500
    max_sequence_length = SEQUENCE_LENGTH
    temperature = 0.4

    lstm_melody = lstm_generator.compose_melody(seed, num_steps, max_sequence_length, temperature)
    print("Melody from LSTM model:")
    print(lstm_melody)
    lstm_generator.save_melody(lstm_melody, filename="lstm_melody.mid")

    gru_melody = gru_generator.compose_melody(seed, num_steps, max_sequence_length, temperature)
    print("Melody from GRU model:")
    print(gru_melody)
    gru_generator.save_melody(gru_melody, filename="gru_melody.mid")

    rnn_melody = rnn_generator.compose_melody(seed, num_steps, max_sequence_length, temperature)
    print("Melody from Vanilla RNN model:")
    print(rnn_melody)
    rnn_generator.save_melody(rnn_melody, filename="rnn_melody.mid")
