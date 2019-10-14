import * as tf from '@tensorflow/tfjs';
import { Callback } from '@tensorflow/tfjs-layers';
import {
  PretrainedDecoderMetadata,
  PretrainedEncoderMetadata,
  Seq2seq,
} from './model/seq2seq';
import { SequenceDecoder } from './sequence-decoder';

export type CallbackArgs = {
  everyEpoch: number;
  seq2seq: Seq2seq;
  pretrainedEncoderMetadata: PretrainedEncoderMetadata;
  pretrainedDecoderMetadata: PretrainedDecoderMetadata;
  testData: string[];
  examplesLength: number;
  targetBeginIndex: number;
};

export class TestBatchCallback extends Callback {
  private sequenceDecoder: SequenceDecoder;
  private readonly everyEpoch: number;
  private readonly encoderModel: tf.LayersModel;
  private readonly decoderModel: tf.LayersModel;
  private readonly testData: string[];
  private readonly examplesLength: number;
  private readonly targetBeginIndex: number;

  constructor(sequenceDecoder: SequenceDecoder, config: CallbackArgs) {
    super();
    this.sequenceDecoder = sequenceDecoder;
    this.targetBeginIndex = config.targetBeginIndex;
    this.testData = config.testData;
    this.everyEpoch = config.everyEpoch;
    this.examplesLength = config.examplesLength;
    this.encoderModel = config.seq2seq.buildPretrainedEncoder(
      config.pretrainedEncoderMetadata,
    );
    this.decoderModel = config.seq2seq.buildPretrainedDecoder(
      config.pretrainedDecoderMetadata,
    );
  }

  async onEpochEnd(epoch: number) {
    if (epoch === 0 || epoch % this.everyEpoch !== 0) {
      return;
    }

    console.warn('Testing values...');
    for (let i = 0; i < this.examplesLength; i++) {
      const sampleIndex = Math.floor(Math.random() * this.testData.length);

      const [inputSentence] = this.testData.slice(sampleIndex, sampleIndex + 1);
      const [targetSentence] = this.testData.slice(
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
      console.log('Input sentence:', inputSentence.trim());
      console.log('Target sentence:', targetSentence.trim());
      console.log('Decoded sentence:', decodedSentence.trim());
    }
  }
}
