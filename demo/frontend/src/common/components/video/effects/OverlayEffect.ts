/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import {hexToRgb} from '@/common/components/video/editor/VideoEditorUtils';
import BaseGLEffect from '@/common/components/video/effects/BaseGLEffect';
import {
  EffectFrameContext,
  EffectInit,
} from '@/common/components/video/effects/Effect';
import vertexShaderSource from '@/common/components/video/effects/shaders/DefaultVert.vert?raw';
import fragmentShaderSource from '@/common/components/video/effects/shaders/Overlay.frag?raw';
import {Tracklet} from '@/common/tracker/Tracker';
import {
  findIndexByTrackletId,
  preAllocateTextures,
} from '@/common/utils/ShaderUtils';
import {RLEObject, decode} from '@/jscocotools/mask';
import invariant from 'invariant';
import {CanvasForm} from 'pts';
import Logger from '@/common/logger/Logger';

export default class OverlayEffect extends BaseGLEffect {
  private _numMasks: number = 0;
  private _numMasksUniformLocation: WebGLUniformLocation | null = null;

  private _masksTextureUnitStart: number = 1;
  private _maskTextures: WebGLTexture[] = [];
  private _clickPosition: number[] | null = null;
  private _activeMask: number = 0;

  private static readonly MAX_MASKS: number = 15;

  constructor() {
    super(8);
    this.vertexShaderSource = vertexShaderSource;
    this.fragmentShaderSource = fragmentShaderSource;
  }

  protected setupUniforms(
    gl: WebGL2RenderingContext,
    program: WebGLProgram,
    init: EffectInit,
  ): void {
    super.setupUniforms(gl, program, init);

    this._numMasksUniformLocation = gl.getUniformLocation(program, 'uNumMasks');
    gl.uniform1i(this._numMasksUniformLocation, this._numMasks);

    this._maskTextures = preAllocateTextures(gl, OverlayEffect.MAX_MASKS);
    gl.uniform2f(gl.getUniformLocation(program, 'uSize'), init.width, init.height);
  }

  apply(form: CanvasForm, context: EffectFrameContext, _tracklets: Tracklet[]) {
    const gl = this._gl;
    const program = this._program;

    invariant(gl !== null, 'WebGL2 context is required');
    invariant(program !== null, 'Not WebGL program found');

    const maxTextureUnits = gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS);
    const requiredUnits = 1 + context.masks.length;
    if (requiredUnits > maxTextureUnits) {
      Logger.warn(`Texture units exceeded: ${requiredUnits} > ${maxTextureUnits}. Limiting to ${maxTextureUnits - 1} masks.`);
    }
    if (context.masks.length > OverlayEffect.MAX_MASKS) {
      Logger.warn(`Mask count ${context.masks.length} exceeds MAX_MASKS ${OverlayEffect.MAX_MASKS}. Truncating.`);
    }

    gl.clearColor(0.0, 0.0, 0.0, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const opacity = [0.5, 0.75, 0.35, 0.95][this.variant % 4];
    gl.uniform1f(
      gl.getUniformLocation(program, 'uTime'),
      context.timeParameter ?? 1.5,
    );
    gl.uniform1f(gl.getUniformLocation(program, 'uOpacity'), opacity);
    gl.uniform1i(
      gl.getUniformLocation(program, 'uBorder'),
      this.variant % this.numVariants < 4 ? 1 : 0,
    );

    if (context.actionPoint) {
      const clickPos = [
        context.actionPoint.position[0] / context.width,
        context.actionPoint.position[1] / context.height,
      ];
      this._clickPosition = clickPos;
      this._activeMask = findIndexByTrackletId(
        context.actionPoint.objectId,
        _tracklets,
      );
    }

    gl.uniform2fv(
      gl.getUniformLocation(program, 'uClickPos'),
      this._clickPosition ?? [0, 0],
    );
    gl.uniform1i(
      gl.getUniformLocation(program, 'uActiveMask'),
      this._activeMask,
    );

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._frameTexture);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA,
      context.width,
      context.height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      context.frame,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

    const numMasks = Math.min(
      context.masks.length,
      OverlayEffect.MAX_MASKS,
      maxTextureUnits - 1,
    );
    gl.uniform1i(this._numMasksUniformLocation, numMasks);

    context.masks.slice(0, numMasks).forEach((mask, index) => {
      const decodedMask = decode([mask.bitmap as RLEObject]);
      const maskData = decodedMask.data as Uint8Array;
      gl.activeTexture(gl.TEXTURE0 + this._masksTextureUnitStart + index);
      gl.bindTexture(gl.TEXTURE_2D, this._maskTextures[index]);

      gl.uniform1i(
        gl.getUniformLocation(program, `uMaskTexture${index}`),
        this._masksTextureUnitStart + index,
      );

      const color = hexToRgb(context.maskColors[index]);
      gl.uniform4f(
        gl.getUniformLocation(program, `uMaskColor${index}`),
        color.r,
        color.g,
        color.b,
        color.a,
      );

      gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.LUMINANCE,
        context.height,
        context.width,
        0,
        gl.LUMINANCE,
        gl.UNSIGNED_BYTE,
        maskData,
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    });

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

    gl.bindTexture(gl.TEXTURE_2D, null);
    context.masks.slice(0, numMasks).forEach((_, index) => {
      gl.activeTexture(gl.TEXTURE0 + this._masksTextureUnitStart + index);
      gl.bindTexture(gl.TEXTURE_2D, null);
    });

    const ctx = form.ctx;
    invariant(this._canvas !== null, 'canvas is required');
    ctx.drawImage(this._canvas, 0, 0);
    this._clickPosition = null;
  }

  async cleanup(): Promise<void> {
    super.cleanup();

    if (this._gl != null) {
      this._maskTextures.forEach(texture => {
        if (texture != null && this._gl != null) {
          this._gl.deleteTexture(texture);
        }
      });
      this._maskTextures = [];
    }
  }
}