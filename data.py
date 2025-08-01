from lyricsgenius import Genius
import re


def get_all_albums(artist_id):
    # Returns a dictionary {album name: album id}
    all_albums = {}
    page = 1 
    while True:
        try:
            genius_album_dict = genius.artist_albums(artist_id, per_page=50, page=page)
            if not genius_album_dict['albums']:
                break
            else:
                for album in genius_album_dict['albums']:
                    album_title = album['name']
                    album_id = album['id']
                    all_albums[album_title] = album_id
                page += 1
        except:
            break
    return all_albums


def get_albums(albums, all_albums):
    # Narrows album selection
    album_ids = {}
    for album in albums:
        album_ids[album] = all_albums[album]
    return album_ids


def get_songs(album_ids):
    # Returns a dictionary {album name: {song name: song id}}
    album_songs = {}
    for album, ID in album_ids.items():
        songs = {}
        for i in range(len(genius.album_tracks(ID)['tracks'])):
            song = genius.album_tracks(ID)['tracks'][i]['song']['title']
            song = re.sub(r"[^\w\s]", "", song) 
            songs[song] = genius.album_tracks(ID)['tracks'][i]['song']['id']
        album_songs[album] = songs
    return album_songs


def clean_dict(album_dict):
    # Removes song repeats and cleans names
    for album, songs in album_dict.items():
        clean_songs = {}
        for song, id in songs.items():
            if "Remix" in song or "Prologue" in song:
                continue
            song = song.replace("\u200b", "")
            song = re.sub(r"[^\w\s']", '', song)
            clean_songs[song] = id
        album_dict[album] = clean_songs
    return album_dict


def get_lyrics(album_songs):
    # Returns a dictionary {album name: {song name: lyrics}}
    for album, songs in album_songs.items():
        for song, ID in songs.items():
            # To circumvent request time outs
            while True:
                try:
                    lyrics = genius.lyrics(ID)
                    break
                except:
                    pass
            lyrics = lyrics[lyrics.find('[V'):]
            lyrics = lyrics.lower()
            lyrics = re.sub(r"\[.*?\]", "", lyrics)
            lyrics = re.sub(r"\d+", "", lyrics)
            lyrics = re.sub(r"[^\w\s']", "", lyrics)
            lyrics = re.sub(r"\n+", "\n", lyrics)
            lyrics = lyrics.replace("\u200b", "")
            lyrics = lyrics.strip()
            songs[song] = lyrics
    return album_songs


def write_csv(album_songs):
    with open("data.csv", "w") as out:
        lines = []
        for i, (album, songs) in enumerate(album_songs.items()):
            for i, (song, lyrics) in enumerate(songs.items()):
                for line in lyrics.split("\n"):
                    if line not in lines:
                        lines.append(line)
                        out.write(f"{album},{song},{line}\n")
    
# Instantiate Genius object with access token
genius = Genius('MZTdkAOEve4ct-6hT0W1Kbzdz2MQr1Nugp1R8uRkc2_Hq0bHRUgh5APmovdNeZqS')

# Get artist ID for Taylor Swift
artist = genius.search_artist('Taylor Swift', max_songs = 1)
taylor_id = artist.id
# taylor_id = 1177

# Get dictionary of album names with their songs and song ids
taylor_all_albums = get_all_albums(taylor_id)
# Gave too many variants so hardcoded album names
taylor_albums = ["Taylor Swift", "Fearless (Taylor’s Version)", "Speak Now (Taylor’s Version)", "Red (Taylor’s Version)", "1989 (Taylor’s Version)", "reputation", "Lover", "folklore (deluxe version)", "evermore (deluxe version)", "Midnights (The Til Dawn Edition)", "THE TORTURED POETS DEPARTMENT: THE ANTHOLOGY"]

# Get album ids from masterlist
taylor_album_ids = get_albums(taylor_albums, taylor_all_albums)

# Get replace album ids with song names and song ids
taylor_album_songs = get_songs(taylor_album_ids)

# Clean repeat songs as best as possible
taylor_album_songs = clean_dict(taylor_album_songs)
# taylor_album_songs = {'Taylor Swift': {'Tim McGraw': 132077, 'Picture to Burn': 132079, 'Teardrops On My Guitar': 57189, 'A Place In This World': 132080, 'Cold as You': 132082, 'The Outside': 132091, 'Tied Together with a Smile': 132092, 'Stay Beautiful': 132093, 'Shouldve Said No': 132094, 'Marys Song Oh My My My': 132095, 'Our Song': 132097, 'Im Only Me When Im With You': 132098, 'Invisible': 132100, 'A Perfectly Good Heart': 132101, 'Teardrops on My Guitar Pop Version': 2244685}, 'Fearless (Taylor’s Version)': {'Fearless Taylors Version': 6478825, 'Fifteen Taylors Version': 6478830, 'Love Story Taylors Version': 6478833, 'Hey Stephen Taylors Version': 6478837, 'White Horse Taylors Version': 6478838, 'You Belong With Me Taylors Version': 6478841, 'Breathe Taylors Version': 6478843, 'Tell Me Why Taylors Version': 6478845, 'Youre Not Sorry Taylors Version': 6478847, 'The Way I Loved You Taylors Version': 6478850, 'Forever  Always Taylors Version': 6478851, 'The Best Day Taylors Version': 6478852, 'Change Taylors Version': 6478853, 'Jump Then Fall Taylors Version': 6478856, 'Untouchable Taylors Version': 6478858, 'Forever  Always Piano Version Taylors Version': 6478860, 'Come In With The Rain Taylors Version': 6478862, 'Superstar Taylors Version': 6478871, 'The Other Side of the Door Taylors Version': 6478873, 'Today Was a Fairytale Taylors Version': 6478877, 'You All Over Me Taylors Version From the Vault': 2975370, 'Mr Perfectly Fine Taylors Version From the Vault': 4499981, 'We Were Happy Taylors Version From the Vault': 187445, 'Thats When Taylors Version From the Vault': 2973355, 'Dont You Taylors Version From the Vault': 4499926, 'Bye Bye Baby Taylors Version From the Vault': 187271}, 'Speak Now (Taylor’s Version)': {'Mine Taylors Version': 6688213, 'Sparks Fly Taylors Version': 6688225, 'Back To December Taylors Version': 6688226, 'Speak Now Taylors Version': 6688227, 'Dear John Taylors Version': 6688228, 'Mean Taylors Version': 6688229, 'The Story Of Us Taylors Version': 6688230, 'Never Grow Up Taylors Version': 6688231, 'Enchanted Taylors Version': 6688232, 'Better Than Revenge Taylors Version': 6688233, 'Innocent Taylors Version': 6688235, 'Haunted Taylors Version': 6688236, 'Last Kiss Taylors Version': 6688237, 'Long Live Taylors Version': 6688238, 'Ours Taylors Version': 6688372, 'Superman Taylors Version': 6688374, 'Electric Touch Taylors Version From The Vault': 9199814, 'When Emma Falls in Love Taylors Version From The Vault': 9199809, 'I Can See You Taylors Version From The Vault': 4499979, 'Castles Crumbling Taylors Version From The Vault': 4499925, 'Foolish One Taylors Version From The Vault': 3947017, 'Timeless Taylors Version From The Vault': 9199321}, 'Red (Taylor’s Version)': {'State of Grace Taylors Version': 6688185, 'Red Taylors Version': 6688196, 'Treacherous Taylors Version': 6688197, 'I Knew You Were Trouble Taylors Version': 6688198, 'All Too Well Taylors Version': 6688199, '22 Taylors Version': 6688200, 'I Almost Do Taylors Version': 6688201, 'We Are Never Ever Getting Back Together Taylors Version': 6688202, 'Stay Stay Stay Taylors Version': 6688203, 'The Last Time Taylors Version': 6688204, 'Holy Ground Taylors Version': 6688205, 'Sad Beautiful Tragic Taylors Version': 6688206, 'The Lucky One Taylors Version': 6688207, 'Everything Has Changed Taylors Version': 6688208, 'Starlight Taylors Version': 6688209, 'Begin Again Taylors Version': 6688210, 'The Moment I Knew Taylors Version': 6688338, 'Come BackBe Here Taylors Version': 6688341, 'Girl At Home Taylors Version': 6688339, 'State Of Grace Acoustic Version Taylors Version': 7379117, 'Ronan Taylors Version': 7078171, 'Better Man Taylors Version From The Vault': 7076619, 'Nothing New Taylors Version From The Vault': 4809175, 'Babe Taylors Version From The Vault': 7076620, 'Message In A Bottle Taylors Version From The Vault': 7078185, 'I Bet You Think About Me Taylors Version From the Vault': 4499978, 'Forever Winter Taylors Version From The Vault': 7076622, 'Run Taylors Version From The Vault': 7076623, 'The Very First Night Taylors Version From The Vault': 7076625, 'All Too Well 10 Minute Version Taylors Version From The Vault': 7076626, 'A Message From Taylor': 7291673}, '1989 (Taylor’s Version)': {'Welcome To New York Taylors Version': 6688259, 'Blank Space Taylors Version': 6688260, 'Style Taylors Version': 6688261, 'Out Of The Woods Taylors Version': 6688262, 'All You Had To Do Was Stay Taylors Version': 6688263, 'Shake It Off Taylors Version': 6688264, 'I Wish You Would Taylors Version': 6688265, 'Bad Blood Taylors Version': 6688266, 'Wildest Dreams Taylors Version': 6688267, 'How You Get The Girl Taylors Version': 6688268, 'This Love Taylors Version': 6688270, 'I Know Places Taylors Version': 6688271, 'Clean Taylors Version': 6688272, 'Wonderland Taylors Version': 6688274, 'You Are In Love Taylors Version': 6688275, 'New Romantics Taylors Version': 6688276, 'Slut Taylors Version From The Vault': 9539416, 'Say Dont Go Taylors Version From The Vault': 9538405, 'Now That We Dont Talk Taylors Version From The Vault': 9666585, 'Suburban Legends Taylors Version From The Vault': 9538403, 'Is It Over Now Taylors Version From The Vault': 9538404}, 'reputation': {'Ready for It': 3221550, 'End Game': 3281773, 'I Did Something Bad': 3306083, 'Dont Blame Me': 3306086, 'Delicate': 3283025, 'Look What You Made Me Do': 3210592, 'So It Goes': 3306084, 'Gorgeous': 3280165, 'Getaway Car': 3283019, 'King of My Heart': 3305019, 'Dancing With Our Hands Tied': 3306080, 'Dress': 3281502, 'This Is Why We Cant Have Nice Things': 3283031, 'Call It What You Want': 3300013, 'New Years Day': 3281524}, 'Lover': {'I Forgot That You Existed': 4782856, 'Cruel Summer': 4712978, 'Lover': 4508914, 'The Man': 4765529, 'The Archer': 4721067, 'I Think He Knows': 4765979, 'Miss Americana  The Heartbreak Prince': 4765976, 'Paper Rings': 4782857, 'Cornelia Street': 4765965, 'Death By A Thousand Cuts': 4782863, 'London Boy': 4782854, 'Soon Youll Get Better': 4765961, 'False God': 4782865, 'You Need To Calm Down': 4625737, 'Afterglow': 4782869, 'ME': 4472545, 'Its Nice To Have A Friend': 4782870, 'Daylight': 4765964}, 'folklore (deluxe version)': {'the 1': 5794073, 'cardigan': 5793984, 'the last great american dynasty': 5793985, 'exile': 5793983, 'my tears ricochet': 5793982, 'mirrorball': 5793981, 'seven': 5793979, 'august': 5793977, 'this is me trying': 5793975, 'illicit affairs': 5793974, 'invisible string': 5793972, 'mad woman': 5793964, 'epiphany': 5793963, 'betty': 5793962, 'peace': 5793961, 'hoax': 5793957, 'the lakes': 5793971, 'folklore Foreword': 6083654}, 'evermore (deluxe version)': {'willow': 6260155, 'champagne problems': 6260160, 'gold rush': 6260161, 'tis the damn season': 6260163, 'tolerate it': 6260164, 'no body no crime': 6260165, 'happiness': 6260166, 'dorothea': 6260167, 'coney island': 6260168, 'ivy': 6260153, 'cowboy like me': 6260173, 'long story short': 6260174, 'marjorie': 6260158, 'closure': 6260162, 'evermore': 6260141, 'right where you left me': 6263242, 'its time to go': 6260178, 'Evermore Liner Notes': 6284890}, 'Midnights (The Til Dawn Edition)': {'Lavender Haze': 8442190, 'Maroon': 8485907, 'AntiHero': 8434253, 'Snow On The Beach': 8445376, 'Youre On Your Own Kid': 8485908, 'Midnight Rain': 8485905, 'Question': 8485912, 'Vigilante Shit': 8485914, 'Bejeweled': 8485915, 'Labyrinth': 8445366, 'Karma': 8485918, 'Sweet Nothing': 8485919, 'Mastermind': 8400683, 'The Great War': 8486441, 'Bigger Than The Whole Sky': 8486443, 'Paris': 8486438, 'High Infidelity': 8486442, 'Glitch': 8486446, 'Wouldve Couldve Shouldve': 8486447, 'Dear Reader': 8486436, 'Hits Different': 8461583, 'Snow On The Beach feat More Lana Del Rey': 9157481}, 'THE TORTURED POETS DEPARTMENT: THE ANTHOLOGY': {'Fortnight': 10024009, 'The Tortured Poets Department': 10024578, 'My Boy Only Breaks His Favorite Toys': 10024528, 'Down Bad': 10024535, 'So Long London': 10024536, 'But Daddy I Love Him': 10024520, 'Fresh Out The Slammer': 10024544, 'Florida': 10291434, 'Guilty as Sin': 10024517, 'Whos Afraid of Little Old Me': 10024563, 'I Can Fix Him No Really I Can': 10024518, 'loml': 10024526, 'I Can Do It With a Broken Heart': 10024512, 'The Smallest Man Who Ever Lived': 10024519, 'The Alchemy': 10024521, 'Clara Bow': 10024516, 'The Black Dog': 10124160, 'imgonnagetyouback': 10296670, 'The Albatross': 10090426, 'Chloe or Sam or Sophia or Marcus': 10296695, 'How Did It End': 10296686, 'So High School': 10296673, 'I Hate It Here': 10296676, 'thanK you aIMee': 10296677, 'I Look in Peoples Windows': 10296681, 'The Prophecy': 10296661, 'Cassandra': 10296680, 'Peter': 10296682, 'The Bolter': 10064067, 'Robin': 10296690, 'The Manuscript': 10021428}}

# Replace song ids with song lyrics
taylor_album_songs = get_lyrics(taylor_album_songs)

# Write to csv file
write_csv(taylor_album_songs)