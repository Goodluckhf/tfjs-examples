import * as tf from '@tensorflow/tfjs';
import { Callback } from '@tensorflow/tfjs-layers';
import { SequenceDecoder } from './sequence-decoder';

export type CallbackArgs = {
  everyEpoch: number;
  testInputData: string[][];
  testTargetData: string[][];
  examplesLength: number;
  targetBeginIndex: number;
};

export class TestBatchCallback extends Callback {
  private sequenceDecoder: SequenceDecoder;
  private readonly everyEpoch: number;
  private readonly encoderModel: tf.LayersModel;
  private readonly decoderModel: tf.LayersModel;
  private readonly examplesLength: number;
  private readonly targetBeginIndex: number;
  private readonly testInputData: string[][];
  private readonly testTargetData: string[][];

  constructor(
    sequenceDecoder: SequenceDecoder,
    encoderModel: tf.LayersModel,
    decoderModel: tf.LayersModel,
    config: CallbackArgs,
  ) {
    super();
    this.sequenceDecoder = sequenceDecoder;
    this.targetBeginIndex = config.targetBeginIndex;
    this.testInputData = config.testInputData;
    this.testTargetData = config.testTargetData;
    this.everyEpoch = config.everyEpoch;
    this.examplesLength = config.examplesLength;
    this.encoderModel = encoderModel;
    this.decoderModel = decoderModel;
  }

  async onEpochEnd(epoch: number) {
    if (epoch === 0 || epoch % this.everyEpoch !== 0) {
      return;
    }

    console.warn('Testing values...');
    for (let i = 0; i < this.examplesLength; i++) {
      const sampleIndex = Math.floor(Math.random() * this.testInputData.length);

      const [inputSentence] = this.testInputData.slice(
        sampleIndex,
        sampleIndex + 1,
      );
      const [targetSentence] = this.testTargetData.slice(
        sampleIndex,
        sampleIndex + 1,
      );
      const { encoderInputs: x } = this.sequenceDecoder.getXSample(
        inputSentence,
        targetSentence,
      );

      // Array to string
      const decodedSentence = await this.sequenceDecoder.decode(
        x.expandDims(),
        this.encoderModel,
        this.decoderModel,
        this.targetBeginIndex,
      );

      console.log('-');
      console.log('Input sentence:', inputSentence.join(' '));
      console.log('Target sentence:', targetSentence.join(' '));
      console.log('Decoded sentence:', decodedSentence.join(' '));
    }
  }
}
