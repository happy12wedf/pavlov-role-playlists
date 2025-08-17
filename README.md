# THIS IS BROKEN RIGHT NOW PLEASE WAIT TILL I FUCKING FIX IT.



# Pavlov Role Playlists

Automatically opens Spotify playlists based on your Pavlov role icons. Detects roles on your screen and launches the corresponding playlist in the Spotify desktop app.

## Features

* Detects Pavlov roles in real-time using screen capture.
* Opens the corresponding Spotify playlist in your desktop app.
* Prevents repeated playlist openings for the same role.
* Easy to customize with your own icons and Spotify playlists.
* Works on Windows, macOS, and Linux.

## Installation

1. Download or extract the folder named `pavlov role thing`.
2. All dependencies are included in the folder, including `opencv-python`, `numpy`, `pyautogui`, and `Pillow` for image handling.
3. Make sure you have **Spotify installed** on your desktop.

## Setup

1. Add your role icons to the `icons/` folder. Make sure the filenames match the keys in the script (e.g., `innocent.png`, `zombie.png`, etc.).
2. Replace the Spotify URIs in the script with your playlist URIs:

```python
ROLES = {
    "innocent": ("icons/innocent.png", "spotify:playlist:YOUR_INNOCENT_URI"),
    "zombie":   ("icons/zombie.png",   "spotify:playlist:YOUR_ZOMBIE_URI"),
    ...
}
```

3. Optionally, set a **screen region** for faster detection by modifying `ROI`:

```python
ROI = (x, y, width, height)  # Example: ROI = (0, 0, 800, 600)
```

## Usage

Run the script from the folder:

```bash
python music.py
```

The script will continuously detect your Pavlov role icons and open the corresponding playlist in Spotify whenever your role changes.

Press `Ctrl+C` to stop the script.

## Notes

* The script only opens playlists once per role change to prevent multiple launches.
* Works entirely locally; does not interact with the game or network.
* Dependencies are included in the folder; no additional installs needed.
* Tested on Windows, macOS, and Linux.
* This should not ban you, EAC detects tampering so that's why this is used with screen reading.

## Contributing

Feel free to submit issues or pull requests for improvements.

## License

nonw
