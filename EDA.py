import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Load dataframe from csv
lyrics_array = np.loadtxt('data.csv', delimiter=",", dtype=str)
lyrics_df = pd.DataFrame(lyrics_array, columns=['Album', 'Song', 'Lyric'])

# Get a list of words form all lyrics
words = []
for i in range(np.shape(lyrics_array)[0]):
    lyrics = lyrics_array[i, 2].lower()
    words.extend(lyrics.split())

# Get basic stats on words
word_counts = Counter(words)
high_freq_words = word_counts.most_common()
unique_words = set(words)
longest_words = sorted(unique_words, key=len, reverse=True)
print(f"Overall:\nHighest Freq. Words: {high_freq_words[:20]}\nLongest words:){longest_words[:20]}")

# Same thing but by album
taylor_albums = ["Taylor Swift", "Fearless (Taylor’s Version)", "Speak Now (Taylor’s Version)", "Red (Taylor’s Version)", "1989 (Taylor’s Version)", "reputation", "Lover", "folklore (deluxe version)", "evermore (deluxe version)", "Midnights (The Til Dawn Edition)", "THE TORTURED POETS DEPARTMENT: THE ANTHOLOGY"]
album_words = [[] for i in range(11)]
for i, album in enumerate(taylor_albums):
    lyrics = lyrics_df.loc[lyrics_df['Album'] == album]['Lyric'].tolist()
    lyrics = [lyric.split() for lyric in lyrics]
    album_words[i] = [word.lower() for line in lyrics for word in line]
for i in range(len(album_words)):
    word_counts = Counter(album_words[i])
    high_freq_words = word_counts.most_common()
    unique_words = set(album_words[i])
    longest_words = sorted(unique_words, key=len, reverse=True)
    print(f"{taylor_albums[i]}:\nHighest Freq. Words: {high_freq_words[:20]}\nLongest words:){longest_words[:20]}")
    plt.bar([x[0] for x in high_freq_words[:10]], [x[1] for x in high_freq_words[:10]])
    plt.title(taylor_albums[i])
    plt.xlabel("Word")
    plt.ylabel("Frequency")
    plt.savefig(f"plots/{taylor_albums[i].replace(' ', '_')}_words.png")
    plt.close()
