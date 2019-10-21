import * as tf from '@tensorflow/tfjs';
import { Layer, LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { LayerVariable } from '@tensorflow/tfjs';
import { nameScope } from '@tensorflow/tfjs-layers/dist/common';

export interface BaseDenseAttentionArgs extends LayerArgs {
  units: number;
}

export class LongauAttention extends Layer {
  private readonly w1: tf.layers.Layer;
  private readonly w2: tf.layers.Layer;
  private readonly v: tf.layers.Layer;

  constructor(args: BaseDenseAttentionArgs) {
    super(args);
    this.w1 = tf.layers.dense({
      units: args.units,
      name: 'AttentionW1',
    });
    this.w2 = tf.layers.dense({
      units: args.units,
      name: 'AttentionW2',
    });
    this.v = tf.layers.dense({
      units: 1,
      name: 'AttentionV',
    });
  }

  build(inputShape: Array<number | null> | tf.Shape[]): void {
    nameScope('LongauAttention_w1', () => {
      this.w1.build(inputShape);
    });
    nameScope('LongauAttention_w2', () => {
      this.w2.build(inputShape);
    });
    nameScope('LongauAttention_v', () => {
      // @ts-ignore
      this.v.build([null, inputShape[inputShape.length - 1]]);
    });
  }

  get trainableWeights(): LayerVariable[] {
    if (!this.trainable) {
      return [];
    }
    // Porting Note: In TypeScript, `this` is always an instance of `Layer`.
    return [...this.w1.trainableWeights, ...this.w2.trainableWeights, ...this.v.trainableWeights];
  }

  get nonTrainableWeights(): LayerVariable[] {
    // Porting Note: In TypeScript, `this` is always an instance of `Layer`.
    if (!this.trainable) {
      return [...this.w1.weights, ...this.w2.weights, ...this.v.weights];
    }
    return [...this.w1.nonTrainableWeights, ...this.w2.nonTrainableWeights, ...this.v.nonTrainableWeights];
  }

  call(inputs: tf.Tensor[]) {
    const [decoderOutput, encoderOutput] = inputs;
    if (!decoderOutput.shape[1]) {
      throw new Error('shape');
    }

    const shaped = decoderOutput.reshape([
      decoderOutput.shape[0],
      1,
      decoderOutput.shape[1],
    ]);

    const score = this.v.call(
      tf.tanh(
        (this.w1.call(encoderOutput, {}) as tf.Tensor).add(this.w2.call(
          shaped,
          {},
        ) as tf.Tensor),
      ),
      {},
    ) as tf.Tensor;

    const attentionWeights = tf.softmax(score);
    const contextVector = attentionWeights.mul(encoderOutput).sum(1);
    return contextVector;
  }

  getClassName(): string {
    return 'LongauAttention';
  }

  /** @nocollapse */
  static className = 'LongauAttention';
}

tf.serialization.registerClass(LongauAttention);
