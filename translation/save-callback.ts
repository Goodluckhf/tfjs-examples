import { Callback, LayersModel } from '@tensorflow/tfjs-layers';
import * as tf from '@tensorflow/tfjs';

export class SaveCallback extends Callback {
  private readonly path: string;
  private readonly everyEpoch: number;
  private readonly encoderModel: LayersModel;
  private readonly decoderModel: LayersModel;

  constructor(
    everyEpoch = 1,
    path: string,
    encoderModel: tf.LayersModel,
    decoderModel: tf.LayersModel,
  ) {
    super();
    this.path = path;
    this.everyEpoch = everyEpoch;
    this.encoderModel = encoderModel;
    this.decoderModel = decoderModel;
  }

  async onEpochEnd(epoch: number) {
    if (epoch === 0 || epoch % this.everyEpoch !== 0) {
      return;
    }
    console.warn('Saving model...');
    await Promise.all([
      this.model.save(this.path, {trainableOnly: true, includeOptimizer: true}),
      this.encoderModel.save(`${this.path}/encoder`, {trainableOnly: true}),
      this.decoderModel.save(`${this.path}/decoder`,{trainableOnly: true}),
    ]);
  }
}
