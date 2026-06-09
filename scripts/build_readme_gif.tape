# Regenerate the README terminal demo GIF.
#
# Requires: vhs (https://github.com/charmbracelet/vhs), ca on PATH, sample map.
# Run via: scripts/build_readme_gif.sh

Output docs/images/readme-demo.gif

Set Shell "bash"
Set FontSize 16
Set Width 960
Set Height 540
Set Theme "Catppuccin Mocha"
Set Padding 24
Set Framerate 30
Set TypingSpeed 50ms

Hide
Type "cd /tmp/cloudanalyzer-readme-gif"
Enter
Sleep 500ms
Show

Type "# compare a downsampled map against the original"
Sleep 800ms
Enter
Sleep 400ms

Type "ca downsample map.pcd -o down.pcd -v 0.2 --evaluate"
Enter
Sleep 4s

Type "# open a static browser viewer with heatmap"
Sleep 800ms
Enter
Sleep 400ms

Type "ca web-export map.pcd down.pcd --heatmap -o /tmp/ca-readme-viewer"
Enter
Sleep 3s
