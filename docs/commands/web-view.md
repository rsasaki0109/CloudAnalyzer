# `ca web-view`

Open results in the [CloudAnalyzer web app](https://rsasaki0109.github.io/CloudAnalyzer/app/) with one
link: the maps and trajectories that `ca posegraph-fix`, `ca posegraph-compare` or any other command
wrote, to look at, measure and fix by hand.

```bash
ca web-view fixed/                 # a folder: its .ply, .pcd, .las/.laz, .e57, .xyz, .obj/.stl, .tum and .splat files
ca web-view fixed/map.ply run.tum  # or files
```

The web app runs in the browser and cannot read files on your disk by itself, so `ca web-view`
serves the files named (only those, on 127.0.0.1 only) and opens the app with one `?url=` per file.
It keeps serving until Ctrl-C. KITTI pose text (`.txt`) is left out of folders: the app would take it
for a point cloud; open it in the app's Trajectories panel instead.

| Option | Default | |
|---|---|---|
| `--port` | any free port | Serve on this port |
| `--app URL` | the published app | Open another deployment (e.g. `http://localhost:5173/` while developing) |
| `--no-browser` | off | Print the link only |

Chrome and Edge may ask once whether the app may reach devices on your local network: allow it,
since that is this machine.

For AI agents, the MCP tool `view_link` ([`ca mcp`](mcp.md)) does the same inside the MCP server and
returns the link to hand to the person.
