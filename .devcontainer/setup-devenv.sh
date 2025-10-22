#!/bin/bash

echo "Setting up development environment..."

# Install tmux and other useful tools
apt update && apt install -y tmux git

# Create tmux configuration
cat > ~/.tmux.conf << 'EOF'
# Enable mouse mode (tmux 2.1 and above)
set -g mouse on

# Enable scrolling with mouse wheel
bind -n WheelUpPane if-shell -F -t = "#{mouse_any_flag}" "send-keys -M" "if -Ft= '#{pane_in_mode}' 'send-keys -M' 'select-pane -t=; copy-mode -e; send-keys -M'"
bind -n WheelDownPane select-pane -t= \; send-keys -M

# Increase scrollback buffer size
set -g history-limit 10000

# Use vi-style key bindings in copy mode
setw -g mode-keys vi

# Make scrolling more responsive
set -g mouse on
set -s escape-time 0

# Better key bindings
bind | split-window -h
bind - split-window -v

# Easy pane switching with Alt+arrow
bind -n M-Left select-pane -L
bind -n M-Right select-pane -R
bind -n M-Up select-pane -U
bind -n M-Down select-pane -D
EOF

echo "Tmux configuration installed successfully!"

# Clone bench-ARK repository if it doesn't exist
BENCH_ARK_DIR="/workspaces/cellpose-ARK/bench-ARK"
if [ ! -d "$BENCH_ARK_DIR" ]; then
    echo "Cloning bench-ARK repository..."
    cd /workspaces/cellpose-ARK
    git clone https://github.com/alexschroeter/bench-ARK.git
    if [ $? -eq 0 ]; then
        echo "bench-ARK repository cloned successfully to $BENCH_ARK_DIR"
    else
        echo "Failed to clone bench-ARK repository"
    fi
else
    echo "bench-ARK repository already exists at $BENCH_ARK_DIR"
    # Optionally update the repository
    echo "Updating bench-ARK repository..."
    cd "$BENCH_ARK_DIR"
    git pull origin main 2>/dev/null || git pull origin master 2>/dev/null || echo "Failed to update repository (no internet or different default branch)"
fi

# Install bench-ARK package
echo "Installing bench-ARK package..."

# Force upgrade setuptools to support PEP 660 editable installs (Docker container, no isolation)
# echo "Upgrading setuptools for PEP 660 support..."
# pip install --upgrade pip
# pip install --upgrade "setuptools>=64.0.0" --root-user-action=ignore --force-reinstall || echo "Failed to upgrade setuptools"

# Try editable install first, fall back to regular install if it fails
if ! pip install --break-system-packages -e $BENCH_ARK_DIR; then
    echo "Editable install failed, trying regular install..."
else
    echo "bench-ARK package installed successfully (editable install)"
fi

# Go back to the original workspace
cd /workspaces/cellpose-ARK

echo "Development environment setup complete!"