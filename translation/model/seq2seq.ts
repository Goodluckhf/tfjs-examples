import * as tf from '@tensorflow/tfjs';
import { LSTM } from '@tensorflow/tfjs-layers/dist/layers/recurrent';
import { AttentionLstm } from '../attention-lstm';

export type Seq2seqArgs = {
  numEncoderTokens: number;
  numDecoderTokens: number;
  inputSequenceLength: number;
  latentDim: number;
  embeddingDim: number;
};

export type PretrainedEncoderMetadata = {
  inputs: tf.SymbolicTensor;
  outputs: tf.SymbolicTensor[];
};

export type PretrainedDecoderMetadata = {
  decoder: {
    inputs: tf.SymbolicTensor;
    outputs: tf.SymbolicTensor;
    embeddingInputs: tf.SymbolicTensor;
    lstm: AttentionLstm;
    softmax: tf.layers.Layer;
  };
};

export class Seq2seq {
  private readonly numEncoderTokens: number;
  private readonly numDecoderTokens: number;

  private readonly inputSequenceLength: number;

  private readonly latentDim: number;
  private readonly embeddingDim: number;

  constructor({
    numEncoderTokens,
    numDecoderTokens,
    inputSequenceLength,
    latentDim,
    embeddingDim,
  }: Seq2seqArgs) {
    this.numEncoderTokens = numEncoderTokens;
    this.numDecoderTokens = numDecoderTokens;
    this.inputSequenceLength = inputSequenceLength;
    this.latentDim = latentDim;
    this.embeddingDim = embeddingDim;
  }

  buildPretrainedEncoder(pretrained: PretrainedEncoderMetadata) {
    return tf.model({
      inputs: pretrained.inputs,
      outputs: pretrained.outputs,
      name: 'pretrainedEncoderModel',
    });
  }

  buildPretrainedDecoder({ decoder }: PretrainedDecoderMetadata) {
    const stateInputH = tf.layers.input({
      shape: [this.latentDim * 2],
      name: 'decoderStateInputHidden',
    });

    const stateInputC = tf.layers.input({
      shape: [this.latentDim * 2],
      name: 'decoderStateInputCell',
    });

    const encoderOutputInput = tf.layers.input({
      shape: [null, this.latentDim * 2],
      name: 'encoderOutputInput',
    });

    const statesInputs = [stateInputH, stateInputC];
    let [sequenceOutput, stateH, stateC] = decoder.lstm.apply([
      decoder.embeddingInputs,
      encoderOutputInput,
      ...statesInputs,
    ]) as tf.SymbolicTensor[];

    const states = [stateH, stateC];
    const outputs = decoder.softmax.apply(sequenceOutput) as tf.SymbolicTensor;

    return tf.model({
      inputs: [decoder.inputs, encoderOutputInput, ...statesInputs],
      outputs: [outputs, ...states],
      name: 'pretrainedDecoderModel',
    });
  }

  buildEncoder() {
    // Define an input sequence and process it.
    const inputs = tf.layers.input({
      shape: [this.inputSequenceLength] as number[],
      name: 'encoderInputs',
    });

    const embeddingInputs = tf.layers
      .embedding({
        inputDim: this.numEncoderTokens,
        outputDim: this.embeddingDim,
        name: 'encoderEmbeddings',
      })
      .apply(inputs);

    const [outputs, forwardH, forwardC, backwardH, backwardC] = tf.layers
      .bidirectional({
        layer: tf.layers.lstm({
          units: this.latentDim,
          returnState: true,
          returnSequences: true,
          name: 'encoderLSTM',
        }) as LSTM,
        name: 'biLSTM',
      })
      .apply(embeddingInputs) as tf.SymbolicTensor[];

    const stateH = tf.layers
      .concatenate()
      .apply([forwardH, backwardH]) as tf.SymbolicTensor;
    const stateC = tf.layers
      .concatenate()
      .apply([forwardC, backwardC]) as tf.SymbolicTensor;

    return {
      states: [stateH, stateC],
      inputs,
      outputs,
    };
  }

  buildDecoder(
    encoderStates: tf.SymbolicTensor[],
    encoderOutputs: tf.SymbolicTensor,
  ): PretrainedDecoderMetadata {
    const inputs = tf.layers.input({
      shape: [null],
      name: 'decoderInputs',
    });

    const embeddingInputs = tf.layers
      .embedding({
        inputDim: this.numDecoderTokens,
        maskZero: true,
        outputDim: this.embeddingDim,
        name: 'decoderEmbedding',
      })
      .apply(inputs) as tf.SymbolicTensor;

    // We set up our decoder to return full output sequences,
    // and to return internal states as well. We don't use the
    // return states in the training model, but we will use them in inference.
    const lstm = new AttentionLstm({
      units: this.latentDim * 2,
      returnSequences: true,
      returnState: true,
      name: 'decoderLSTM',
    }) as AttentionLstm;

    const [decoderOutputs] = lstm.apply([
      embeddingInputs,
      encoderOutputs,
      ...encoderStates,
    ]) as tf.SymbolicTensor[];


    const softmax = tf.layers.dense({
      units: this.numDecoderTokens,
      activation: 'softmax',
      name: 'decoderSoftmax',
    });

    const denseOutputs = softmax.apply(decoderOutputs) as tf.SymbolicTensor;

    return {
      decoder: {
        inputs,
        outputs: denseOutputs,
        embeddingInputs,
        lstm,
        softmax,
      },
    };
  }
}
