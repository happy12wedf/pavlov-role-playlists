import cv2
import numpy as np
import pyautogui
import time
import sys
import json
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth

MATCH_THRESHOLD = 0.92
POLL_INTERVAL = 0.2
CONFIDENCE_FRAMES = 2
HIGH_CONFIDENCE_THRESHOLD = 0.96
ROI = None
CONFIG_FILE = "spotify_config.json"

def load_config():
    default_config = {
        "client_id": "",
        "client_secret": "",
        "redirect_uri": "http://127.0.0.1:8080/callback"
    }
    
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    else:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(default_config, f, indent=4)
        print(f"created {CONFIG_FILE}. add your app stuff")
        return default_config

def setup_spotify():
    config = load_config()
    
    if not config["client_id"] or not config["client_secret"]:
        print("configure your settings in pavlov_spotify_config.json")
        print("1. go to https://developer.spotify.com/dashboard")
        print("2. create a new app")
        print("3. set redirect uri to: http://127.0.0.1:8080/callback")
        print("4. copy client id and secret to pavlov_spotify_config.json")
        return None
    
    try:
        scope = "user-modify-playback-state user-read-playback-state"
        auth_manager = SpotifyOAuth(
            client_id=config["client_id"],
            client_secret=config["client_secret"],
            redirect_uri=config["redirect_uri"],
            scope=scope,
            cache_path=".spotify_cache"
        )
        return spotipy.Spotify(auth_manager=auth_manager)
    except Exception as e:
        print(f"failed: {e}")
        return None

ROLES = {
    "innocent":   ("icons/innocent.png",   "spotify:playlist:YOUR_INNOCENT_URI"),
    "soulmate":   ("icons/soulmate.png",   "spotify:playlist:YOUR_SOULMATE_URI"),
    "jester":     ("icons/jester.png",     "spotify:playlist:YOUR_JESTER_URI"),
    "psychopath": ("icons/psychopath.png", "spotify:playlist:YOUR_PSYCHOPATH_URI"),
    "zombie":     ("icons/zombie.png",     "spotify:playlist:5iNepJl1wuY6Fw4eyJXV0z"),
    "detective":  ("icons/detective.png",  "spotify:playlist:YOUR_DETECTIVE_URI"),
    "sheriff":    ("icons/sheriff.png",    "spotify:playlist:YOUR_SHERIFF_URI"),
    "tank":       ("icons/tank.png",       "spotify:playlist:YOUR_TANK_URI"),
    "traitor":    ("icons/traitor.png",    "spotify:playlist:YOUR_TRAITOR_URI"),
    "hypnotist":  ("icons/hypnotist.png",  "spotify:playlist:YOUR_HYPNOTIST_URI"),
    "assassin":   ("icons/assassin.png",   "spotify:playlist:YOUR_ASSASSIN_URI"),
    "mercenary":  ("icons/mercenary.png",  "spotify:playlist:YOUR_MERCENARY_URI"),
    "lonewolf":   ("icons/LoneWolf.png",   "spotify:playlist:YOUR_LONEWOLF_URI"),
}

def load_templates():
    templates = {}
    for role, (icon_path, uri) in ROLES.items():
        img = cv2.imread(icon_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        templates[role] = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), uri)
    return templates

def screenshot_gray():
    pyautogui.FAILSAFE = False
    shot = pyautogui.screenshot(region=ROI) if ROI else pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def detect_role(gray_frame, templates):
    best_match = None
    best_score = 0
    best_uri = None
    
    for role, (tmpl, uri) in templates.items():
        if gray_frame.shape[0] < tmpl.shape[0] or gray_frame.shape[1] < tmpl.shape[1]:
            continue
        res = cv2.matchTemplate(gray_frame, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > best_score and max_val >= MATCH_THRESHOLD:
            best_match = role
            best_score = max_val
            best_uri = uri
    
    if best_match:
        print(f"  -> {best_match}: {best_score:.3f} confidence")
        return best_match, best_uri, best_score
    return None, None, 0

def play_spotify_playlist(spotify, uri):
    if not spotify:
        print("spotify not configured")
        return False
    
    try:
        devices = spotify.devices()
        if not devices["devices"]:
            print("no spotify devices found")
            return False
        
        device_id = devices["devices"][0]["id"]
        device_name = devices["devices"][0]["name"]
        
        if uri.startswith("spotify:playlist:"):
            playlist_id = uri.replace("spotify:playlist:", "")
        else:
            playlist_id = uri
            
        context_uri = f"spotify:playlist:{playlist_id}"
        
        spotify.start_playback(
            device_id=device_id,
            context_uri=context_uri
        )
        
        print(f"playing {device_name}")
        return True
        
    except Exception as e:
        print(f"failed to start the playlist {e}")
        return False

def main():
    print("pavlov spotify controller")
    print("=" * 50)
    
    spotify = setup_spotify()
    if not spotify:
        print("failure to connect to spotify")
        return
    
    try:
        user = spotify.current_user()
        print(f"connected as: {user['display_name']}")
    except Exception as e:
        print(f"connection failed: {e}")
        return
    
    templates = load_templates()
    if not templates:
        print("no roles in icons folder")
        print("please re add the icons cause they come with the repo whered they go")
        return
    
    print(f"loaded {len(templates)} role templates")
    print("starting scanning")
    print("press ctrl+c to stop")
    print()
    
    
    current_role = None
    current_playlist = None
    role_detections = {}
    detection_count = 0
    
    try:
        while True:
            detection_count += 1
            if detection_count % 10 == 1:
                print(f"scan #{detection_count}...")
            
            gray = screenshot_gray()
            role, uri, confidence = detect_role(gray, templates)
            
            if role:
                if role not in role_detections:
                    role_detections[role] = 0
                
                role_detections[role] += 1
                
                for other_role in list(role_detections.keys()):
                    if other_role != role:
                        role_detections[other_role] = 0
                
                should_switch = False
                switch_reason = ""
                
                if role != current_role and uri != current_playlist:
                    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                        should_switch = True
                        switch_reason = f"high confidence ({confidence:.3f})"
                    elif role_detections[role] >= CONFIDENCE_FRAMES:
                        should_switch = True
                        switch_reason = f"confirmed ({role_detections[role]} detections)"
                
                if should_switch:
                    print(f"\nrole: {role.upper()} - {switch_reason}")
                    if play_spotify_playlist(spotify, uri):
                        current_role = role
                        current_playlist = uri
                        print(f"switched to {role} playlist")
                        role_detections[role] = 0
                    else:
                        print("failed to change list")
                    print()
                    
            else:
                # only clears if no role is seen for a long time
                # this prevents resetting when 1 scan is missed
                all_zero = all(count == 0 for count in role_detections.values())
                if not all_zero:
                    for role_key in role_detections:
                        if role_detections[role_key] > 0:
                            role_detections[role_key] = max(0, role_detections[role_key] - 1)
                
                if all(count == 0 for count in role_detections.values()) and current_role:
                    print("no role detected")
                    current_role = None
                    current_playlist = None
                
            time.sleep(POLL_INTERVAL)
            
    except KeyboardInterrupt:
        print("\nstopping")
        print("ty for using")

if __name__ == "__main__":
    main()
