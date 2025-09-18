<p align="center">
  <img width="423" alt="Screenshot" src="https://github.com/user-attachments/assets/a2df02d5-db58-40a8-8904-c3663483841e" />
</p>

This is an [ISF shader](https://isf.video) for simulating a cell system,
converted from [this ShaderToy shader](https://www.shadertoy.com/view/3tSfRW) by
[**@MichaelMoroz**](https://github.com/MichaelMoroz).

This is a multi-pass shader that is intended to be used with floating-point
buffers. Not all ISF hosts support floating-point buffers.
[Videosync](https://videosync.showsync.com/download) supports floating-point
buffers in
[v2.0.12](https://support.showsync.com/release-notes/videosync/2.0#2012) and
later, but https://editor.isf.video does not appear to support floating-point
buffers. This shader will produce *very* different output if floating-point
buffers are not used.
