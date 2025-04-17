import matplotlib.pyplot as plt
import parselmouth

class Audio_files():
    def __init__(self, audio1):
        self.audio1 = audio1

    def compute_contour(self, label:str):
        aud1 = parselmouth.Sound((self.audio1))
        pitch1 = aud1.to_pitch()
        time1 = pitch1.xs()
        values1= pitch1.selected_array['frequency']
        plt.title("Audio1")
        plt.xlabel("Time", loc="left")
        plt.ylabel("F0")
        plt.plot(time1, values1, label=label)


    def voice_segments(self):

        sound1 = parselmouth.Sound(self.audio1)
        # Load the audio file
        pitch = sound1.to_pitch(pitch_floor = 75,
                                pitch_ceiling = 500)

        # Get time and frequency values
        times = pitch.xs()
        frequencies = pitch.selected_array['frequency']

        # Voiced intervals (where frequency > 0)
        voiced_segments = []
        current_start = None

        for time, frequency in zip(times, frequencies):
            if frequency > 0:  # Voiced frame
                if current_start is None:
                    current_start = time  # Start of voiced segment
            else:  # Unvoiced frame
                if current_start is not None:
                    voiced_segments.append((current_start, time))  # End of segment
                    current_start = None

        # If the last frame was voiced, close the segment
        if current_start is not None:
            voiced_segments.append((current_start, times[-1]))

        # Print the voiced intervals
        for start, end in voiced_segments:
            print(f"Voiced interval: {start:.2f} - {end:.2f} seconds")
