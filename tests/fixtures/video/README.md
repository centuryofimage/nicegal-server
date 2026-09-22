# Generated video fixtures

These small test-pattern videos were generated with FFmpeg's `testsrc2` source.
They contain no user media. FFmpeg's CLI is needed only to regenerate the fixtures,
not to run tests or the app.

```sh
ffmpeg -f lavfi -i 'testsrc2=size=128x96:rate=10:duration=6' -an -c:v libx264 -threads 1 -g 20 -keyint_min 20 -sc_threshold 0 -pix_fmt yuv420p -metadata creation_time=2020-01-02T03:04:05Z keyframes.mp4
ffmpeg -display_rotation 90 -i keyframes.mp4 -c copy rotated.mp4
ffmpeg -f lavfi -i 'testsrc2=size=128x96:rate=10:duration=2' -an -c:v libx265 -threads 1 -x265-params 'pools=none:frame-threads=1:keyint=10:log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc' -pix_fmt yuv420p10le -color_primaries bt2020 -color_trc smpte2084 -colorspace bt2020nc hevc10.mp4
ffmpeg -f lavfi -i 'testsrc2=size=64x48:rate=10:duration=3' -f lavfi -i 'sine=frequency=440:sample_rate=16000:duration=8' -c:v libx264 -g 10 -pix_fmt yuv420p -c:a aac -b:a 32k short-video-long-container.mkv
```
