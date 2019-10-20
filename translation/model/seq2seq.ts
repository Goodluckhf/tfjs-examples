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

export type PretrainedAttentionMetadata = {
  attentionSoftmax: tf.layers.Layer;
  attentionDot: tf.layers.Layer;
  contextDot: tf.layers.Layer;
  contextConcatenate: tf.layers.Layer;
  tanhDense: tf.layers.Layer;
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

  pretrainedAttention(
    {
      attentionSoftmax,
      attentionDot,
      contextDot,
      contextConcatenate,
      tanhDense,
    }: PretrainedAttentionMetadata,
    encoderOutput: tf.SymbolicTensor,
    decoderOutput: tf.SymbolicTensor,
  ) {
    const attention = attentionSoftmax.apply(
      attentionDot.apply([decoderOutput, encoderOutput]),
    ) as tf.SymbolicTensor;

    const context = contextDot.apply([
      attention,
      encoderOutput,
    ]) as tf.SymbolicTensor;

    const decoderCombinedContext = contextConcatenate.apply([
      context,
      decoderOutput,
    ]);

    return tanhDense.apply(decoderCombinedContext);
  }

  attention(
    encoderOutput: tf.SymbolicTensor,
    decoderOutput: tf.SymbolicTensor,
    lstmUnits: number,
  ) {
    const attentionDot = tf.layers.dot({ axes: [2, 2] });
    const attentionSoftmax = tf.layers.activation({
      activation: 'softmax',
      name: 'Attention',
    });
    const contextDot = tf.layers.dot({
      axes: [2, 1],
      name: 'context',
    });
    const contextConcatenate = tf.layers.concatenate();
    const tanhDense = tf.layers.timeDistributed({
      layer: tf.layers.dense({
        units: lstmUnits,
        activation: 'tanh',
      }),
    });

    const attention = attentionSoftmax.apply(
      attentionDot.apply([decoderOutput, encoderOutput]),
    ) as tf.SymbolicTensor;

    const context = contextDot.apply([
      attention,
      encoderOutput,
    ]) as tf.SymbolicTensor;

    const decoderCombinedContext = contextConcatenate.apply([
      context,
      decoderOutput,
    ]);

    const outputs = tanhDense.apply(decoderCombinedContext);

    return {
      attentionDot,
      attentionSoftmax,
      contextDot,
      contextConcatenate,
      tanhDense,
      outputs,
    };
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
    let [sequenceOutput, stateH, stateC] = decoder.lstm.apply(
      // @ts-ignore
      [decoder.embeddingInputs, encoderOutputInput, ...statesInputs],
      // {
      //   initialState: statesInputs,
      // },
    ) as tf.SymbolicTensor[];

    // const attentionLayer = this.pretrainedAttention(
    //   attention,
    //   encoderOutputInput,
    //   sequenceOutput,
    // );

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

    // @ts-ignore
    const [decoderOutputs] = lstm.apply(
      [embeddingInputs, encoderOutputs, ...encoderStates],
      {
        //initialState: encoderStates,
      },
    ) as tf.SymbolicTensor[];

    // const { outputs: attentionOutput, ...attentionLayers } = this.attention(
    //   encoderOutputs,
    //   decoderOutputs,
    //   this.latentDim,
    // );

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
