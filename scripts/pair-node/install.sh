#!/usr/bin/env bash
# Make this dev box a PAIR node that is there for Edinburgh's 3 AM ET analysis.
#
# 1. PAIR (the desktop app, /opt/PAIR/nvpair) starts with the desktop session,
#    so the node is back in the cluster after a login or reboot.
# 2. A system timer wakes the box from suspend at 02:50 ET on weekdays
#    (needs sudo once). GNOME's idle suspend on AC is off on this box, so it
#    stays up until you suspend it yourself.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

install -D -m 644 "$here/nvpair-autostart.desktop" ~/.config/autostart/nvpair.desktop
echo "autostart: ~/.config/autostart/nvpair.desktop"

sudo install -m 644 "$here/edinburgh-wake.service" "$here/edinburgh-wake.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now edinburgh-wake.timer
systemctl list-timers edinburgh-wake.timer --no-pager
