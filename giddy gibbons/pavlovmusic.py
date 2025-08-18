import cv2
import numpy as np
import pyautogui
import time
import sys
import json
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
import threading
import gc

MATCH_THRESHOLD = 15
POLL_INTERVAL = 0.15
CONFIDENCE_FRAMES = 4
HIGH_CONFIDENCE_THRESHOLD = 25
ROI = None
MIN_MATCH_COUNT = 8
AKAZE_THRESHOLD = 0.65
CONTEXT_CHECK = False
MIN_CONFIDENCE_GAP = 8
DETECTION_MEMORY_FRAMES = 8
STABILITY_THRESHOLD = 0.8
USE_COLOR_DISCRIMINATION = False
JESTER_PENALTY = 3
MIN_FEATURE_DENSITY = 0.02
ICON_PRESENCE_THRESHOLD = 0.3
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
    "innocent":   ("icons/innocent/",   "spotify:playlist:YOUR_INNOCENT_URI"),
    "jester":     ("icons/jester/",     "spotify:playlist:YOUR_JESTER_URI"),
    "zombie":     ("icons/zombie/",     "spotify:playlist:YOUR_ZOMBIE_URI"),
    "sheriff":    ("icons/sheriff/",    "spotify:playlist:YOUR_SHERIFF_URI"),
    "traitor":    ("icons/traitor/",    "spotify:playlist:YOUR_TRAITOR_URI"),
    "hypnotist":  ("icons/hypnotist/",  "spotify:playlist:YOUR_HYPNO_URI"),
    "assassin":   ("icons/assassin/",   "spotify:playlist:YOUR_ASSASSIN_URI"),
    "lonewolf":   ("icons/lonewolf/",   "spotify:playlist:YOUR_LONEWOLF_URI"),
    "preround":   ("icons/preround/",   "spotify:playlist:YOUR_PREROUND_URI"),
    "mercenary":  ("icons/mercenary/",  "spotify:playlist:YOUR_MERCENARY_URI"),
    "tank":       ("icons/tank/",       "spotify:playlist:YOUR_TANK_URI"),
    "detective":  ("icons/detective/",  "spotify:playlist:YOUR_DETECTIVE_URI"),
}

AKAZE_DETECTOR = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, 
                                   descriptor_size=0, descriptor_channels=3, 
                                   threshold=0.001, nOctaves=4, nOctaveLayers=4, 
                                   diffusivity=cv2.KAZE_DIFF_PM_G2)
MATCHER = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
CLAHE = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

def load_templates():
    akaze = AKAZE_DETECTOR
    templates = {}
    
    for role, (folder_path, uri) in ROLES.items():
        if not os.path.exists(folder_path):
            print(f"warning: folder {folder_path} not found")
            continue
        
        all_keypoints = []
        all_descriptors = []
        image_count = 0
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                
                if img is None:
                    continue
                
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                enhanced = CLAHE.apply(gray)
                
                blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
                
                keypoints, descriptors = akaze.detectAndCompute(blurred, None)
                
                if descriptors is not None and len(keypoints) > 0:
                    all_keypoints.extend(keypoints)
                    if len(all_descriptors) == 0:
                        all_descriptors = descriptors
                    else:
                        all_descriptors = np.vstack([all_descriptors, descriptors])
                    image_count += 1
        
        if len(all_descriptors) > 0 and len(all_keypoints) >= MIN_MATCH_COUNT:
            templates[role] = {
                'keypoints': all_keypoints,
                'descriptors': all_descriptors,
                'uri': uri,
                'image_count': image_count
            }
            print(f"loaded {role}: {len(all_keypoints)} AKAZE features from {image_count} images")
        else:
            print(f"warning: {role} has too few AKAZE features ({len(all_keypoints)} from {image_count} images)")
    
    return templates

def is_spotify_ui(gray_frame):
    try:
        height, width = gray_frame.shape
        
        bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        green_pixels = cv2.countNonZero(green_mask)
        total_pixels = height * width
        green_ratio = green_pixels / total_pixels
        
        if green_ratio > 0.01:
            return True
        
        top_region = gray_frame[0:int(height*0.3), :]
        dark_pixels = cv2.countNonZero(top_region < 50)
        dark_ratio = dark_pixels / (top_region.shape[0] * top_region.shape[1])
        
        if dark_ratio > 0.6:
            return True
        
        return False
    except Exception as e:
        print(f"spotify ui check error: {e}")
        return False

def is_pavlov_context(gray_frame):
    try:
        height, width = gray_frame.shape
        
        if is_spotify_ui(gray_frame):
            return False
        
        bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = height * width
        red_ratio = red_pixels / total_pixels
        
        if red_ratio > 0.005:
            return True
        
        bottom_region = gray_frame[int(height*0.7):height, :]
        
        text_like_features = cv2.countNonZero(cv2.Canny(bottom_region, 50, 150))
        
        if text_like_features > 100:
            return True
        
        return False
    except Exception as e:
        print(f"context check error: {e}")
        return True

def screenshot_gray():
    pyautogui.FAILSAFE = False
    shot = pyautogui.screenshot(region=ROI) if ROI else pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def has_role_icon_features(gray_frame, keypoints):
    if len(keypoints) < 5:
        return False
    
    height, width = gray_frame.shape
    
    feature_density = len(keypoints) / (height * width / 1000)
    if feature_density < MIN_FEATURE_DENSITY:
        return False
    
    center_x, center_y = width // 2, height // 2
    icon_region_features = 0
    
    for kp in keypoints:
        x, y = kp.pt
        dist_from_center = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        relative_dist = dist_from_center / min(width, height)
        
        if relative_dist < ICON_PRESENCE_THRESHOLD:
            icon_region_features += 1
    
    icon_feature_ratio = icon_region_features / len(keypoints)
    return icon_feature_ratio > 0.2

def get_role_color_signature(gray_frame):
    try:
        bgr_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        
        red_mask1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255]))
        red_mask2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
        red_mask = red_mask1 + red_mask2
        
        green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))
        
        purple_mask = cv2.inRange(hsv, np.array([120, 50, 50]), np.array([150, 255, 255]))
        
        total_pixels = gray_frame.shape[0] * gray_frame.shape[1]
        red_ratio = cv2.countNonZero(red_mask) / total_pixels
        green_ratio = cv2.countNonZero(green_mask) / total_pixels
        purple_ratio = cv2.countNonZero(purple_mask) / total_pixels
        
        return {
            'red': red_ratio,
            'green': green_ratio,
            'purple': purple_ratio
        }
    except:
        return {'red': 0, 'green': 0, 'purple': 0}

def detect_role(gray_frame, templates, detection_history=None):
    try:
        enhanced_frame = CLAHE.apply(gray_frame)
        blurred_frame = cv2.GaussianBlur(enhanced_frame, (3, 3), 0)
        
        frame_keypoints, frame_descriptors = AKAZE_DETECTOR.detectAndCompute(blurred_frame, None)
        
        if frame_descriptors is None:
            return None, None, 0
        
        if not has_role_icon_features(gray_frame, frame_keypoints):
            return None, None, 0
        
        color_sig = get_role_color_signature(gray_frame) if USE_COLOR_DISCRIMINATION else None
        
        best_match = None
        best_score = 0
        best_uri = None
        second_best_score = 0
        role_scores = {}
        
        for role, template_data in templates.items():
            try:
                template_descriptors = template_data['descriptors']
                image_count = template_data.get('image_count', 1)
                
                matches = MATCHER.knnMatch(template_descriptors, frame_descriptors, k=2)
                
                good_matches = []
                excellent_matches = []
                
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < AKAZE_THRESHOLD * n.distance:
                            good_matches.append(m)
                            if m.distance < 0.5 * n.distance:
                                excellent_matches.append(m)
                
                raw_match_count = len(good_matches)
                excellent_count = len(excellent_matches)
                
                quality_bonus = excellent_count * 0.4
                
                color_bonus = 0
                color_penalty = 0
                
                if USE_COLOR_DISCRIMINATION and color_sig:
                    if role == 'hypnotist':
                        if color_sig['purple'] > 0.015:
                            color_bonus += 5
                        if color_sig['green'] > 0.025:
                            color_penalty -= 3
                    elif role == 'zombie':
                        if color_sig['green'] > 0.025:
                            color_bonus += 5
                        if color_sig['purple'] > 0.015:
                            color_penalty -= 3
                    elif role in ['traitor', 'assassin']:
                        if color_sig['red'] > 0.025:
                            color_bonus += 4
                        if color_sig['green'] > 0.015:
                            color_penalty -= 2
                    elif role == 'innocent':
                        if color_sig['red'] > 0.04:
                            color_penalty -= 5
                        if color_sig['green'] > 0.04:
                            color_penalty -= 5
                    elif role == 'jester':
                        if color_sig['purple'] > 0.015 or color_sig['red'] > 0.02:
                            color_bonus += 2
                        else:
                            color_penalty -= JESTER_PENALTY
                        if raw_match_count < 15:
                            color_penalty -= 2
                    elif role == 'sheriff':
                        if color_sig['red'] > 0.02:
                            color_bonus += 2
                
                final_bonus = quality_bonus + color_bonus + color_penalty
                normalized_score = (raw_match_count + final_bonus) / max(1, image_count * 0.25)
                
                role_scores[role] = {
                    'raw_matches': raw_match_count,
                    'excellent_matches': excellent_count,
                    'color_bonus': color_bonus,
                    'color_penalty': color_penalty,
                    'normalized_score': normalized_score,
                    'image_count': image_count
                }
                
                if normalized_score > best_score:
                    second_best_score = best_score
                    best_score = normalized_score
                    best_match = role
                    best_uri = template_data['uri']
                elif normalized_score > second_best_score:
                    second_best_score = normalized_score
            except Exception as e:
                print(f"error matching {role}: {e}")
                continue
        
        if best_match and best_score >= MATCH_THRESHOLD * 0.6:
            confidence_gap = best_score - second_best_score
            raw_matches = role_scores[best_match]['raw_matches']
            excellent_matches = role_scores[best_match]['excellent_matches']
            
            stability_bonus = 0
            consistency_bonus = 0
            flickering_penalty = 0
            if detection_history and len(detection_history) > 0:
                recent_detections = detection_history[-5:]
                role_count = recent_detections.count(best_match)
                if role_count >= 3:
                    stability_bonus = 4
                elif role_count >= 2:
                    stability_bonus = 2
                
                if len(recent_detections) >= 4:
                    unique_roles = len(set(recent_detections))
                    if unique_roles <= 2:
                        consistency_bonus = 2
                    elif unique_roles >= 4:
                        flickering_penalty = -3
                
                recent_none_count = recent_detections.count(None)
                if recent_none_count >= 2:
                    flickering_penalty -= 2
            
            adjusted_gap = confidence_gap + stability_bonus + consistency_bonus + flickering_penalty
            
            if adjusted_gap >= MIN_CONFIDENCE_GAP and raw_matches >= MATCH_THRESHOLD:
                color_info = ""
                if USE_COLOR_DISCRIMINATION and best_match in role_scores:
                    cb = role_scores[best_match]['color_bonus']
                    cp = role_scores[best_match]['color_penalty']
                    if cb > 0 or cp < 0:
                        color_info = f", color: {cb:+.1f}{cp:+.1f}"
                
                print(f"  -> {best_match}: {raw_matches} AKAZE matches, {excellent_matches} excellent (norm: {best_score:.2f}, gap: {adjusted_gap:.2f}{color_info})")
                return best_match, best_uri, raw_matches
        
        return None, None, 0
    except Exception as e:
        print(f"detection error: {e}")
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

def test_screenshot(screenshot_path):
    print(f"testing with screenshot: {screenshot_path}")
    
    templates = load_templates()
    if not templates:
        print("no templates loaded")
        return
    
    img = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    if img is None:
        print("could not load screenshot")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    print("testing AKAZE detection...")
    role, uri, match_count = detect_role(gray, templates)
    
    if role:
        print(f"detected: {role} with {match_count} AKAZE feature matches")
    else:
        print("no role detected")
        print("showing match counts for all roles:")
        
        akaze = cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        frame_keypoints, frame_descriptors = akaze.detectAndCompute(blurred, None)
        
        if frame_descriptors is None:
            print("no AKAZE features found in screenshot")
            return
        
        print(f"screenshot has {len(frame_keypoints)} AKAZE features")
        
        for role_name, template_data in templates.items():
            template_descriptors = template_data['descriptors']
            image_count = template_data.get('image_count', 1)
            
            matches = bf.knnMatch(template_descriptors, frame_descriptors, k=2)
            
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < AKAZE_THRESHOLD * n.distance:
                        good_matches.append(m)
            
            raw_matches = len(good_matches)
            normalized_score = raw_matches / max(1, image_count * 0.25)
            print(f"  {role_name}: {raw_matches} AKAZE matches, {normalized_score:.2f} normalized (from {image_count} images)")

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
            role, uri, confidence = detect_role(gray, templates, getattr(detect_role, 'history', []))
            
            if not hasattr(detect_role, 'history'):
                detect_role.history = []
            detect_role.history.append(role)
            if len(detect_role.history) > DETECTION_MEMORY_FRAMES:
                detect_role.history.pop(0)
            
            if role:
                if role not in role_detections:
                    role_detections[role] = 0
                
                role_detections[role] += 1
                
                for other_role in list(role_detections.keys()):
                    if other_role != role:
                        role_detections[other_role] = 0
                
                should_switch = False
                switch_reason = ""
                
                if role != current_role:
                    if confidence >= HIGH_CONFIDENCE_THRESHOLD:
                        should_switch = True
                        switch_reason = f"high confidence ({confidence:.3f})"
                    elif role_detections[role] >= CONFIDENCE_FRAMES:
                        should_switch = True
                        switch_reason = f"confirmed ({role_detections[role]} detections)"
                elif role == current_role:
                    if detection_count % 50 == 1:
                        print(f"maintaining {role} role (scan #{detection_count})")
                
                if should_switch:
                    print(f"\nrole: {role.upper()} - {switch_reason}")
                    if play_spotify_playlist(spotify, uri):
                        current_role = role
                        current_playlist = uri
                        print(f"switched to {role} playlist")
                        role_detections = {role: 0}
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
                
            if detection_count % 20 == 0:
                gc.collect()
            
            time.sleep(POLL_INTERVAL)
            
    except KeyboardInterrupt:
        print("\nstopping")
        print("ty for using")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test" and len(sys.argv) > 2:
        test_screenshot(sys.argv[2])
    else:
        main()
