#!/usr/bin/env bash
# Add "YuE 2 – Music Gen Studio" to the app menu. Opens a terminal running run.sh, so the
# window is the tool's console: close it and the tool stops.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A="$HOME/.local/share/applications"; mkdir -p "$A"
cat > "$A/ai-yue2.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=AI – YuE 2 – Music Gen Studio
Comment=music generation web app (localhost:7863)
Icon=audio-x-generic
Exec=$HERE/run.sh 
Terminal=true
Categories=AudioVideo;
Keywords=AI;generation;
StartupNotify=false
EOF
command -v update-desktop-database >/dev/null && update-desktop-database "$A" || true
echo "installed $A/ai-yue2.desktop"
