import * as tf from '@tensorflow/tfjs';
import { Layer, LayerArgs } from '@tensorflow/tfjs-layers/dist/engine/topology';

export interface BaseDenseAttentionArgs extends LayerArgs {
  units: number;
}

export class LongauAttention extends Layer {
  private readonly wa: tf.layers.Layer;
  private readonly attentionSoftmax: tf.layers.Layer;

  // constructor(args: BaseDenseAttentionArgs) {
  //   super(args);
  //   this.wa = tf.layers.dense({
  //     units: args.units,
  //     name: 'AttentionScore',
  //   });
  // }

  constructor(args: BaseDenseAttentionArgs) {
    super(args);
    this.wa = tf.layers.dense({
      units: args.units,
      name: 'AttentionScore',
    });
    this.attentionSoftmax = tf.layers.softmax({
      axis: 2,
      name: 'Attention',
    });
  }

  build(inputShape: Array<number | null> | tf.Shape[]): void {
    this.wa.build(inputShape);
  }

  // call(inputs: tf.Tensor[]) {
  //   const [decoderOutput, encoderOutput] = inputs;
  //
  //   const score = tf.matMul(
  //     decoderOutput,
  //     // @ts-ignore
  //     this.wa.call(encoderOutput),
  //     false,
  //     true,
  //   );
  //
  //   const alignment = tf.softmax(score, 2);
  //   const context = tf.matMul(alignment, encoderOutput);
  //   return [context, alignment];
  // }

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

    const score = tf.matMul(
      shaped,
      this.wa.call(encoderOutput, {}) as tf.Tensor,
      false,
      true,
    );

    const alignment = this.attentionSoftmax.call(score, {}) as tf.Tensor;

    const context = tf.matMul(alignment, encoderOutput);
    return context;
  }

  getClassName(): string {
    return 'LongauAttention';
  }

  /** @nocollapse */
  static className = 'LongauAttention';
}

tf.serialization.registerClass(LongauAttention);
