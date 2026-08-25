"""Class definitions shared across the project.

Mask pixel values are the integer class ids below; 0 is background.
The submission file names classes by string, so CLASS_LABELS defines the
label -> id mapping used at encode time.
"""

CLASS_NAMES = {
    0: "background",
    1: "hex_nut",
    2: "washer",
    3: "bolt",
    4: "ball_bearing",
    5: "spring",
    6: "o_ring",
}

# Foreground labels in submission order (id = index + 1)
CLASS_LABELS = ["hex_nut", "washer", "bolt", "ball_bearing", "spring", "o_ring"]
CLASS_ID = {name: i + 1 for i, name in enumerate(CLASS_LABELS)}

# RGB colours for mask visualisation
CLASS_COLORS = {
    0: (30, 30, 30),
    1: (255, 80, 80),
    2: (80, 180, 255),
    3: (255, 200, 0),
    4: (0, 220, 120),
    5: (200, 80, 255),
    6: (255, 140, 40),
}

# ImageNet statistics — used for input normalisation even though the encoder is
# trained from scratch, purely to keep activations well-scaled at init.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
