import * as tf from '@tensorflow/tfjs';
import { END_OF_SENTENCE, UNKNOWN } from './constants';

export type SequenceDecoderArgs = {
  inputTokenIndex: { [char: string]: number };
  targetTokenIndex: { [char: string]: number };
  reverseTargetCharIndex: { [indice: number]: string };
  maxDecoderSeqLength: number;
  maxEncoderSeqLength: number;
};

export class SequenceDecoder {
  private readonly reverseTargetCharIndex: { [p: number]: string };
  private readonly inputTokenIndex: { [p: string]: number };
  private readonly maxEncoderSeqLength: number;
  private readonly maxDecoderSeqLength: number;
  private readonly targetTokenIndex: { [p: string]: number };

  constructor(config: SequenceDecoderArgs) {
    this.inputTokenIndex = config.inputTokenIndex;
    this.targetTokenIndex = config.targetTokenIndex;
    this.reverseTargetCharIndex = config.reverseTargetCharIndex;
    this.maxEncoderSeqLength = config.maxEncoderSeqLength;
    this.maxDecoderSeqLength = config.maxDecoderSeqLength;
  }

  getXSample(inputSentence: string[], targetSentence: string[]) {
    return tf.tidy(() => {
      const encoderInputDataBuf = tf.buffer<tf.Rank.R1>([
        this.maxEncoderSeqLength,
      ]);

      const decoderInputDataBuf = tf.buffer<tf.Rank.R1>([
        this.maxDecoderSeqLength,
      ]);
      for (const [t, char] of inputSentence.entries()) {
        encoderInputDataBuf.set(
          this.inputTokenIndex[char] || this.inputTokenIndex[UNKNOWN],
          t,
        );
      }

      for (const [t, char] of targetSentence.entries()) {
        decoderInputDataBuf.set(
          this.targetTokenIndex[char] || this.targetTokenIndex[UNKNOWN],
          t,
        );
      }

      return {
        encoderInputs: encoderInputDataBuf.toTensor(),
        decoderInputs: decoderInputDataBuf.toTensor(),
      };
    });
  }

  getYSample(targetSentence: string[]) {
    return tf.tidy(() => {
      const decoderTargetDataBuf = tf.buffer<tf.Rank.R1>([
        this.maxDecoderSeqLength,
      ]);

      for (const [t, char] of targetSentence.entries()) {
        if (t > 0) {
          decoderTargetDataBuf.set(this.targetTokenIndex[char], t - 1);
        }
      }

      return decoderTargetDataBuf.toTensor();
    });
  }

  async decode(
    inputSeq: tf.Tensor,
    encoderModel: tf.LayersModel,
    decoderModel: tf.LayersModel,
    targetBeginIndex: number,
  ) {
    // Encode the input as state vectors.
    let [sequences, ...statesValue] = encoderModel.predict(
      inputSeq,
    ) as tf.Tensor[];

    // Generate empty target sequence of length 1.
    let targetSeq = tf.buffer<tf.Rank.R2>([1, 1]);

    // Populate the first character of target sequence with the start character.
    targetSeq.set(targetBeginIndex, 0, 0);

    // Sampling loop for a batch of sequences
    // (to simplify, here we assume a batch of size 1).
    let stopCondition = false;
    let decodedSentence: string[] = [];
    while (!stopCondition) {
      const [outputTokens, h, c] = decoderModel.predict(
        [targetSeq.toTensor(), sequences, ...statesValue],
        {
          verbose: true,
        },
      ) as [
        tf.Tensor<tf.Rank.R3>,
        tf.Tensor<tf.Rank.R2>,
        tf.Tensor<tf.Rank.R2>,
      ];
      // Sample a token
      const sampledTokenIndex = (await outputTokens
        .squeeze()
        .argMax(-1)
        .array()) as number;

      const sampledChar = this.reverseTargetCharIndex[sampledTokenIndex];
      decodedSentence.push(sampledChar);

      // Exit condition: either hit max length
      // or find stop character.
      if (
        sampledChar === END_OF_SENTENCE ||
        decodedSentence.length > this.maxDecoderSeqLength
      ) {
        stopCondition = true;
      }

      // Update the target sequence (of length 1).
      targetSeq = tf.buffer<tf.Rank.R2>([1, 1]);
      targetSeq.set(sampledTokenIndex, 0, 0);

      // Update states
      statesValue = [h, c];
    }

    return decodedSentence;
  }
}
