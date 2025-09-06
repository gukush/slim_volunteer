// optional vertex shader for WebGL2 block matmul
#version 300 es
precision highp float;
layout (location=0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }