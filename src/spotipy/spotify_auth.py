import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotify_config import CLIENT_ID, CLIENT_SECRET, REDIRECT_URI, SCOPE

def get_spotify_client():
    auth_manager = SpotifyOAuth(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope=SCOPE,
        cache_path=".spotifycache",
        open_browser=True,        
        show_dialog=True         
    )

    # Intenta refrescar o cargar token
    token_info = auth_manager.get_cached_token()
    if not token_info:
        print(" No hay token. Se abrirá el navegador para iniciar sesión.")
        auth_manager.get_access_token(as_dict=False)

    return spotipy.Spotify(auth_manager=auth_manager)

if __name__ == "__main__":
    sp = get_spotify_client()
    user = sp.current_user()
    print("Autenticado como:", user["display_name"])
