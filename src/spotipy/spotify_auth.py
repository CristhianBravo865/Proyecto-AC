import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotify_config import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, SCOPE

def get_spotify_client():
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=".spotifycache"
    ))
    return sp

if __name__ == "__main__":
    sp = get_spotify_client()
    user = sp.current_user()
    print("Autenticado como:", user["display_name"])
