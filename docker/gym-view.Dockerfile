# v1.0.0 line -- viewing image: the training image plus the X11/Qt system libraries
# a *live window* needs (`./run.sh clearance-watch --mode human`).
#
# Recording an mp4 needs none of this -- it renders offscreen in the base image.
# This layer sits on top of caatc-train so a live window works for both scripted
# baselines and trained models (`--model`), without re-downloading torch.
#
#   ./run.sh gym-build && ./run.sh train-build && ./run.sh view-build
FROM caatc-train

# OpenCV's bundled Qt "xcb" platform plugin (and pygame's SDL) need these; without
# them cv2.imshow fails with 'Could not load the Qt platform plugin "xcb"'.
# fontconfig + a font also silence pygame's 'fc-list is missing' warning and let
# the gym renderer draw its text overlays.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libx11-xcb1 libxcb1 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
      libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-shm0 libxcb-sync1 \
      libxcb-xfixes0 libxcb-xinerama0 libxcb-glx0 libxcb-util1 \
      libxkbcommon-x11-0 libxext6 libxrender1 libsm6 libice6 libdbus-1-3 \
      libfontconfig1 libfreetype6 fontconfig fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

CMD ["python", "-m", "caatc.play", "--help"]
