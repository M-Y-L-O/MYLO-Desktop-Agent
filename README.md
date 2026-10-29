# MYLO Desktop Agent

Local compute agent for MYLO Studio. Default bind: `127.0.0.1:8080`.

Studio auto-probes that address on home load. **Start MYLO Client** in the web UI invokes `mylo://launch` (protocol handler registration on the packaged exe is planned — until then, start with `python App.py`).
