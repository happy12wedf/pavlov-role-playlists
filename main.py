import cv2
import numpy as np
import pyautogui
import time
import subprocess
import sys

MATCH_THRESHOLD = 0.85
POLL_INTERVAL = 0.6
ROI = None

ROLES = {
    "innocent":   ("icons/innocent.png",   "spotify:playlist:YOUR_INNOCENT_URI"),
    "soulmate":   ("icons/soulmate.png",   "spotify:playlist:YOUR_SOULMATE_URI"),
    "jester":     ("icons/jester.png",     "spotify:playlist:YOUR_JESTER_URI"),
    "psychopath": ("icons/psychopath.png", "spotify:playlist:YOUR_PSYCHOPATH_URI"),
    "zombie":     ("icons/zombie.png",     "spotify:playlist:YOUR_ZOMBIE_URI"),
    "detective":  ("icons/detective.png",  "spotify:playlist:YOUR_DETECTIVE_URI"),
    "sheriff":    ("icons/sheriff.png",    "spotify:playlist:YOUR_SHERIFF_URI"),
    "tank":       ("icons/tank.png",       "spotify:playlist:YOUR_TANK_URI"),
    "traitor":    ("icons/traitor.png",    "spotify:playlist:YOUR_TRAITOR_URI"),
    "hypnotist":  ("icons/hypnotist.png",  "spotify:playlist:YOUR_HYPNOTIST_URI"),
    "assassin":   ("icons/assassin.png",   "spotify:playlist:YOUR_ASSASSIN_URI"),
    "mercenary":  ("icons/mercenary.png",  "spotify:playlist:YOUR_MERCENARY_URI"),
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
    shot = pyautogui.screenshot(region=ROI) if ROI else pyautogui.screenshot()
    frame = cv2.cvtColor(np.array(shot), cv2.COLOR_RGB2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def detect_role(gray_frame, templates):
    for role, (tmpl, uri) in templates.items():
        if gray_frame.shape[0] < tmpl.shape[0] or gray_frame.shape[1] < tmpl.shape[1]:
            continue
        res = cv2.matchTemplate(gray_frame, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val >= MATCH_THRESHOLD:
            return role, uri
    return None, None

def open_spotify_playlist(uri):
    if sys.platform.startswith("win"):
        subprocess.Popen(["start", uri], shell=True)
    elif sys.platform.startswith("darwin"):
        subprocess.Popen(["open", uri])
    else:
        subprocess.Popen(["xdg-open", uri])

def main():
    templates = load_templates()
    last_role = None
    while True:
        gray = screenshot_gray()
        role, uri = detect_role(gray, templates)
        if role and role != last_role:
            open_spotify_playlist(uri)
            last_role = role
        elif role is None:
            last_role = None  # reset when no role is detected
        time.sleep(POLL_INTERVAL)

if __name__ == "__main__":
    main()
