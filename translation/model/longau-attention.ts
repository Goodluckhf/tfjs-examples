import * as tf from '@tensorflow/tfjs';
import { Layer, LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

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
    this.v = this.w2 = tf.layers.dense({
      units: 1,
      name: 'AttentionV',
    });
  }

  build(inputShape: Array<number | null> | tf.Shape[]): void {
    this.w1.build(inputShape);
    this.w2.build(inputShape);
    this.v.build([1]);
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
