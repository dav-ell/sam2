#version 300 es
// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

precision highp float;

in vec2 vTexCoord;

uniform sampler2D uSampler;
uniform vec2 uSize;
uniform int uNumMasks;
uniform float uOpacity;
uniform bool uBorder;

#define MAX_MASKS 15
uniform sampler2D uMaskTexture0;
uniform sampler2D uMaskTexture1;
uniform sampler2D uMaskTexture2;
uniform sampler2D uMaskTexture3;
uniform sampler2D uMaskTexture4;
uniform sampler2D uMaskTexture5;
uniform sampler2D uMaskTexture6;
uniform sampler2D uMaskTexture7;
uniform sampler2D uMaskTexture8;
uniform sampler2D uMaskTexture9;
uniform sampler2D uMaskTexture10;
uniform sampler2D uMaskTexture11;
uniform sampler2D uMaskTexture12;
uniform sampler2D uMaskTexture13;
uniform sampler2D uMaskTexture14;

uniform vec4 uMaskColor0;
uniform vec4 uMaskColor1;
uniform vec4 uMaskColor2;
uniform vec4 uMaskColor3;
uniform vec4 uMaskColor4;
uniform vec4 uMaskColor5;
uniform vec4 uMaskColor6;
uniform vec4 uMaskColor7;
uniform vec4 uMaskColor8;
uniform vec4 uMaskColor9;
uniform vec4 uMaskColor10;
uniform vec4 uMaskColor11;
uniform vec4 uMaskColor12;
uniform vec4 uMaskColor13;
uniform vec4 uMaskColor14;

uniform float uTime;
uniform vec2 uClickPos;
uniform int uActiveMask;

out vec4 fragColor;

vec4 lowerSaturation(vec4 color, float saturationFactor) {
  float luminance = 0.299f * color.r + 0.587f * color.g + 0.114f * color.b;
  vec3 gray = vec3(luminance);
  vec3 saturated = mix(gray, color.rgb, saturationFactor);
  return vec4(saturated, color.a);
}

vec4 detectEdges(sampler2D textureSampler, float coverage, vec4 edgeColor) {
  vec2 tvTexCoord = vec2(vTexCoord.y, vTexCoord.x);
  vec2 texOffset = 1.0f / uSize;
  vec3 result = vec3(0.0f);
  vec3 tLeft = texture(textureSampler, tvTexCoord + texOffset * vec2(-coverage, coverage)).rgb;
  vec3 tRight = texture(textureSampler, tvTexCoord + texOffset * vec2(coverage, -coverage)).rgb;
  vec3 bLeft = texture(textureSampler, tvTexCoord + texOffset * vec2(-coverage, -coverage)).rgb;
  vec3 bRight = texture(textureSampler, tvTexCoord + texOffset * vec2(coverage, coverage)).rgb;

  vec3 xEdge = tLeft + 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(-coverage, 0)).rgb + bLeft - tRight - 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(coverage, 0)).rgb - bRight;
  vec3 yEdge = tLeft + 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(0, coverage)).rgb + tRight - bLeft - 2.0f * texture(textureSampler, tvTexCoord + texOffset * vec2(0, -coverage)).rgb - bRight;

  result = sqrt(xEdge * xEdge + yEdge * yEdge);
  return result.r > 1e-6f ? edgeColor : vec4(0.0f, 0.0f, 0.0f, 0.0f);
}

vec2 calculateAdjustedTexCoord(vec2 vTexCoord, vec4 bbox, float aspectRatio) {
  vec2 center = vec2((bbox.x + bbox.z) * 0.5f, bbox.w);
  float radiusX = abs(bbox.z - bbox.x);
  float radiusY = radiusX / aspectRatio;
  float scale = 1.0f;
  radiusX *= scale;
  radiusY *= scale;
  vec2 adjustedTexCoord = (vTexCoord - center) / vec2(radiusX, radiusY) + vec2(0.5f);
  return adjustedTexCoord;
}

void main() {
  vec4 color = texture(uSampler, vTexCoord);
  float saturationFactor = 0.7;
  float aspectRatio = uSize.y / uSize.x;
  vec2 tvTexCoord = vec2(vTexCoord.y, vTexCoord.x);

  vec4 finalColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  float totalMaskValue = 0.0f;
  vec4 edgeColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
  float numRipples = 1.75;
  float timeThreshold = 1.1;
  vec2 adjustedClickCoord = calculateAdjustedTexCoord(vTexCoord, vec4(uClickPos, uClickPos + 0.1), aspectRatio);

  for (int i = 0; i < MAX_MASKS; i++) {
    if (i >= uNumMasks) break;

    float maskValue;
    vec4 maskColor;

    if (i == 0) { maskValue = texture(uMaskTexture0, tvTexCoord).r; maskColor = uMaskColor0; }
    else if (i == 1) { maskValue = texture(uMaskTexture1, tvTexCoord).r; maskColor = uMaskColor1; }
    else if (i == 2) { maskValue = texture(uMaskTexture2, tvTexCoord).r; maskColor = uMaskColor2; }
    else if (i == 3) { maskValue = texture(uMaskTexture3, tvTexCoord).r; maskColor = uMaskColor3; }
    else if (i == 4) { maskValue = texture(uMaskTexture4, tvTexCoord).r; maskColor = uMaskColor4; }
    else if (i == 5) { maskValue = texture(uMaskTexture5, tvTexCoord).r; maskColor = uMaskColor5; }
    else if (i == 6) { maskValue = texture(uMaskTexture6, tvTexCoord).r; maskColor = uMaskColor6; }
    else if (i == 7) { maskValue = texture(uMaskTexture7, tvTexCoord).r; maskColor = uMaskColor7; }
    else if (i == 8) { maskValue = texture(uMaskTexture8, tvTexCoord).r; maskColor = uMaskColor8; }
    else if (i == 9) { maskValue = texture(uMaskTexture9, tvTexCoord).r; maskColor = uMaskColor9; }
    else if (i == 10) { maskValue = texture(uMaskTexture10, tvTexCoord).r; maskColor = uMaskColor10; }
    else if (i == 11) { maskValue = texture(uMaskTexture11, tvTexCoord).r; maskColor = uMaskColor11; }
    else if (i == 12) { maskValue = texture(uMaskTexture12, tvTexCoord).r; maskColor = uMaskColor12; }
    else if (i == 13) { maskValue = texture(uMaskTexture13, tvTexCoord).r; maskColor = uMaskColor13; }
    else if (i == 14) { maskValue = texture(uMaskTexture14, tvTexCoord).r; maskColor = uMaskColor14; }

    maskColor /= 255.0;
    vec4 saturatedColor = lowerSaturation(maskColor, saturationFactor);
    vec4 plainColor = vec4(vec3(saturatedColor).rgb, 1.0);
    vec4 rippleColor = vec4(maskColor.rgb, 0.2);

    if (uActiveMask == i && uTime < timeThreshold) {
      float dist = length(adjustedClickCoord);
      float colorFactor = abs(sin((dist - uTime) * numRipples));
      plainColor = vec4(mix(rippleColor, plainColor, colorFactor));
    }
    if (uTime >= timeThreshold) {
      plainColor = vec4(vec3(saturatedColor).rgb, 1.0);
    }

    finalColor += maskValue * plainColor;
    totalMaskValue += maskValue;

    if (edgeColor.a <= 0.0f) {
      if (i == 0) edgeColor = detectEdges(uMaskTexture0, 1.25, maskColor);
      else if (i == 1) edgeColor = detectEdges(uMaskTexture1, 1.25, maskColor);
      else if (i == 2) edgeColor = detectEdges(uMaskTexture2, 1.25, maskColor);
      else if (i == 3) edgeColor = detectEdges(uMaskTexture3, 1.25, maskColor);
      else if (i == 4) edgeColor = detectEdges(uMaskTexture4, 1.25, maskColor);
      else if (i == 5) edgeColor = detectEdges(uMaskTexture5, 1.25, maskColor);
      else if (i == 6) edgeColor = detectEdges(uMaskTexture6, 1.25, maskColor);
      else if (i == 7) edgeColor = detectEdges(uMaskTexture7, 1.25, maskColor);
      else if (i == 8) edgeColor = detectEdges(uMaskTexture8, 1.25, maskColor);
      else if (i == 9) edgeColor = detectEdges(uMaskTexture9, 1.25, maskColor);
      else if (i == 10) edgeColor = detectEdges(uMaskTexture10, 1.25, maskColor);
      else if (i == 11) edgeColor = detectEdges(uMaskTexture11, 1.25, maskColor);
      else if (i == 12) edgeColor = detectEdges(uMaskTexture12, 1.25, maskColor);
      else if (i == 13) edgeColor = detectEdges(uMaskTexture13, 1.25, maskColor);
      else if (i == 14) edgeColor = detectEdges(uMaskTexture14, 1.25, maskColor);
    }
  }

  if (totalMaskValue > 0.0f) {
    finalColor /= totalMaskValue;
    finalColor = mix(color, finalColor, uOpacity);
  } else {
    finalColor.a = 0.0f;
  }

  if (edgeColor.a > 0.0f && uBorder) {
    finalColor = vec4(vec3(edgeColor), 1.0);
  }
  fragColor = finalColor;
}