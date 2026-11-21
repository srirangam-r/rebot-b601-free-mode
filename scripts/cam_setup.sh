#!/usr/bin/env bash
# Re-apply camera settings (they reset when a camera is unplugged or the laptop reboots).
# Overhead ("Wed Camera"): 60 Hz anti-flicker, manual exposure 16.7 ms (2x 1/120 s), gain 39, auto white balance off.
# Wrist (LRCP G720P): exposes no exposure control; only auto white balance can be turned off.
set -e
# Device paths: pass your own as $1 (overhead) and $2 (wrist); the defaults are the cameras this was built with.
OVER=${1:-/dev/v4l/by-id/usb-Wed_Camera_Wed_Camera_20240420112206-video-index0}
WRIST=${2:-/dev/v4l/by-id/usb-LuoKe_Technology_Co.__Ltd._LRCP_G720P_SN0001-video-index0}
v4l2-ctl -d $OVER -c power_line_frequency=2 -c auto_exposure=1 -c exposure_time_absolute=167 -c gain=39 -c white_balance_automatic=0
v4l2-ctl -d $OVER -C power_line_frequency,auto_exposure,exposure_time_absolute,gain,white_balance_automatic
v4l2-ctl -d $WRIST -C white_balance_automatic
