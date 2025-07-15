# cellpose for Arkitekt

There are different aspects of running cellpose in Arkitekt. Inference is simply done by running the default model or models on your data. Cellpose also offers Human-In-The-Loop training which can be done when exporting the GUI locally via Xforwarding.

## Human in the Loop

```bash
xhost +local:docker
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix cellpose-for-arkitekt
```

## GPU Support

Arkitekt supports different GPU accelerator hardware to achieve better performance with the hardware which is available to you.

