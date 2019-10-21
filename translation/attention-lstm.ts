import * as tf from '@tensorflow/tfjs';
import { Kwargs } from '@tensorflow/tfjs-layers/dist/types';
import {
  LSTMLayerArgs,
  rnn,
  RNNLayerArgs,
  RNNCell,
} from '@tensorflow/tfjs-layers/dist/layers/recurrent';
import { AttentionLstmCell } from './attention-lstm-cell';
import { ValueError } from '@tensorflow/tfjs-layers/dist/errors';
import { LongauAttention } from './model/longau-attention';
import {
  Activation,
  serializeActivation,
} from '@tensorflow/tfjs-layers/dist/activations';
import {
  Initializer,
  serializeInitializer,
} from '@tensorflow/tfjs-layers/dist/initializers';
import {
  Regularizer,
  serializeRegularizer,
} from '@tensorflow/tfjs-layers/dist/regularizers';
import {
  Constraint,
  serializeConstraint,
} from '@tensorflow/tfjs-layers/dist/constraints';
import { Shape } from '@tensorflow/tfjs-layers/dist/keras_format/common';
import { isArrayOfShapes } from '@tensorflow/tfjs-layers/dist/utils/types_utils';
import { InputSpec } from '@tensorflow/tfjs-layers/dist/engine/topology';
import { util } from '@tensorflow/tfjs';

export class AttentionLstm extends tf.layers.Layer {
  /** @nocollapse */
  static className = 'AttentionLstm';
  private readonly attentionLayer: LongauAttention;
  private readonly wc: tf.layers.Layer;
  private cell: RNNCell;
  private readonly returnState: boolean;
  private readonly latentDim: number;
  // @ts-ignore
  private stateSpec: InputSpec[];
  private readonly returnSequences: boolean;

  constructor(args: LSTMLayerArgs) {
    if (args.implementation === 0) {
      console.warn(
        '`implementation=0` has been deprecated, and now defaults to ' +
          '`implementation=1`. Please update your layer call.',
      );
    }
    super(args as RNNLayerArgs);
    this.returnSequences = args.returnSequences || false;
    this.returnState = args.returnState || false;
    this.inputSpec = [
      new InputSpec({ ndim: 3 }),
      new InputSpec({ ndim: 3 }),
      new InputSpec({ ndim: 2 }),
      new InputSpec({ ndim: 2 }),
    ];
    this.cell = new AttentionLstmCell(args);
    this.latentDim = args.units;
    this.attentionLayer = new LongauAttention({ units: args.units });
    this.wc = tf.layers.dense({ units: args.units, activation: 'tanh' });
  }

  computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[] {
    if (isArrayOfShapes(inputShape)) {
      inputShape = (inputShape as Shape[])[0];
    }
    inputShape = inputShape as Shape;

    // TODO(cais): Remove the casting once stacked RNN cells become supported.
    let stateSize = this.cell.stateSize;
    if (!Array.isArray(stateSize)) {
      stateSize = [stateSize];
    }
    const outputDim = stateSize[0];
    let outputShape: Shape | Shape[];
    if (this.returnSequences) {
      outputShape = [inputShape[0], inputShape[1], outputDim];
    } else {
      outputShape = [inputShape[0], outputDim];
    }

    if (this.returnState) {
      const stateShape: Shape[] = [];
      for (const dim of stateSize) {
        stateShape.push([inputShape[0], dim]);
      }
      return [outputShape].concat(stateShape);
    } else {
      return outputShape;
    }
  }

  public build(inputShape: Shape | Shape[]): void {
    // Note inputShape will be an Array of Shapes of initial states and
    // constants if these are passed in apply().
    if (isArrayOfShapes(inputShape)) {
      inputShape = (inputShape as Shape[])[0];
    }
    inputShape = inputShape as Shape;

    // const inputDim = inputShape[inputShape.length - 1];
    // this.inputSpec[0] = new InputSpec({ shape: [batchSize, null, inputDim] });
    // this.inputSpec[1] = new InputSpec({ shape: [batchSize, null, inputDim] });

    // Allow cell (if RNNCell Layer) to build before we set or validate
    // stateSpec.
    const stepInputShape = [inputShape[0]].concat(inputShape.slice(2));
    if (!stepInputShape[1]) {
      throw new Error('stepInputShape');
    }
    // @ts-ignore
    this.cell.build([null, stepInputShape[1] + this.latentDim]);
    this.attentionLayer.build([null, null, this.latentDim]);
    this.wc.build([null, this.latentDim * 2]);
    // Set or validate stateSpec.
    let stateSize: number[];
    if (Array.isArray(this.cell.stateSize)) {
      stateSize = this.cell.stateSize;
    } else {
      stateSize = [this.cell.stateSize];
    }

    if (this.stateSpec != null) {
      if (
        !util.arraysEqual(
          // @ts-ignore
          this.stateSpec.map(spec => spec.shape[spec.shape.length - 1]),
          stateSize,
        )
      ) {
        throw new ValueError(
          `An initialState was passed that is not compatible with ` +
            `cell.stateSize. Received stateSpec=${this.stateSpec}; ` +
            `However cell.stateSize is ${this.cell.stateSize}`,
        );
      }
    } else {
      this.stateSpec = stateSize.map(
        dim => new InputSpec({ shape: [null, dim] }),
      );
    }
    if (this.stateful) {
      this.resetStates();
    }
  }

  call(
    inputs: tf.Tensor | tf.Tensor[],
    kwargs: Kwargs,
  ): tf.Tensor | tf.Tensor[] {
    // Input shape: `[samples, time (padded with zeros), input_dim]`.
    // Note that the .build() method of subclasses **must** define
    // this.inputSpec and this.stateSpec owith complete input shapes.
    return tf.tidy(() => {
      if (this.cell.dropoutMask != null) {
        tf.dispose(this.cell.dropoutMask);
        // @ts-ignore
        this.cell.dropoutMask = null;
      }
      if (this.cell.recurrentDropoutMask != null) {
        tf.dispose(this.cell.recurrentDropoutMask);
        // @ts-ignore
        this.cell.recurrentDropoutMask = null;
      }
      const mask = kwargs == null ? null : (kwargs['mask'] as tf.Tensor);
      const training = kwargs == null ? null : kwargs['training'];
      let initialState: tf.Tensor[] =
        kwargs == null ? null : kwargs['initialState'];

      // @ts-ignore
      const [decoderInput, encoderInputs, ...initialStates] = inputs;

      if (initialState == null) {
        if (this.stateful) {
          // @ts-ignore
          initialState = this.states_;
        } else {
          initialState = initialStates;
        }
      }

      const numStates = Array.isArray(this.cell.stateSize)
        ? this.cell.stateSize.length
        : 1;
      if (initialState.length !== numStates) {
        throw new ValueError(
          `RNN Layer has ${numStates} state(s) but was passed ` +
            `${initialState.length} initial state(s).`,
        );
      }

      const cellCallKwargs: Kwargs = { training };

      // TODO(cais): Add support for constants.
      const step = (cellInputs: tf.Tensor, states: tf.Tensor[]) => {
        // `inputs` and `states` are concatenated to form a single `Array` of
        // `tf.Tensor`s as the input to `cell.call()`.
        const context = this.attentionLayer.call([states[0], encoderInputs]);
        const inputWithAttention = tf.concat([context, cellInputs], -1);

        const outputs = this.cell.call(
          [inputWithAttention].concat(states),
          cellCallKwargs,
        ) as tf.Tensor[];
        // Marshall the return value into output and new states.
        return [outputs[0], outputs.slice(1)] as [tf.Tensor, tf.Tensor[]];
      };

      // TODO(cais): Add support for constants.

      const rnnOutputs = rnn(
        step,
        // @ts-ignore
        decoderInput,
        initialState,
        false,
        // @ts-ignore
        mask,
        null,
        false,
        true,
      );
      const lastOutput = rnnOutputs[0];
      const outputs = rnnOutputs[1];
      const states = rnnOutputs[2];

      const output = true ? outputs : lastOutput;

      // TODO(cais): Porperty set learning phase flag.

      if (this.returnState) {
        return [output].concat(states);
      } else {
        return output;
      }
    });
  }

  get units(): number {
    return (this.cell as AttentionLstmCell).units;
  }

  get activation(): Activation {
    return (this.cell as AttentionLstmCell).activation;
  }

  get recurrentActivation(): Activation {
    return (this.cell as AttentionLstmCell).recurrentActivation;
  }

  get useBias(): boolean {
    return (this.cell as AttentionLstmCell).useBias;
  }

  get kernelInitializer(): Initializer {
    return (this.cell as AttentionLstmCell).kernelInitializer;
  }

  get recurrentInitializer(): Initializer {
    return (this.cell as AttentionLstmCell).recurrentInitializer;
  }

  get biasInitializer(): Initializer {
    return (this.cell as AttentionLstmCell).biasInitializer;
  }

  get unitForgetBias(): boolean {
    // @ts-ignore
    return (this.cell as AttentionLstmCell).unitForgetBias;
  }

  get kernelRegularizer(): Regularizer {
    return (this.cell as AttentionLstmCell).kernelRegularizer;
  }

  get recurrentRegularizer(): Regularizer {
    return (this.cell as AttentionLstmCell).recurrentRegularizer;
  }

  get biasRegularizer(): Regularizer {
    return (this.cell as AttentionLstmCell).biasRegularizer;
  }

  get kernelConstraint(): Constraint {
    return (this.cell as AttentionLstmCell).kernelConstraint;
  }

  get recurrentConstraint(): Constraint {
    return (this.cell as AttentionLstmCell).recurrentConstraint;
  }

  get biasConstraint(): Constraint {
    return (this.cell as AttentionLstmCell).biasConstraint;
  }

  get dropout(): number {
    return (this.cell as AttentionLstmCell).dropout;
  }

  get recurrentDropout(): number {
    return (this.cell as AttentionLstmCell).recurrentDropout;
  }

  get implementation(): number {
    return (this.cell as AttentionLstmCell).implementation;
  }

  getConfig(): tf.serialization.ConfigDict {
    const config: tf.serialization.ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      recurrentActivation: serializeActivation(this.recurrentActivation),
      useBias: this.useBias,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      recurrentInitializer: serializeInitializer(this.recurrentInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      unitForgetBias: this.unitForgetBias,
      kernelRegularizer: serializeRegularizer(this.kernelRegularizer),
      recurrentRegularizer: serializeRegularizer(this.recurrentRegularizer),
      biasRegularizer: serializeRegularizer(this.biasRegularizer),
      activityRegularizer: serializeRegularizer(this.activityRegularizer),
      kernelConstraint: serializeConstraint(this.kernelConstraint),
      recurrentConstraint: serializeConstraint(this.recurrentConstraint),
      biasConstraint: serializeConstraint(this.biasConstraint),
      dropout: this.dropout,
      recurrentDropout: this.recurrentDropout,
      implementation: this.implementation,
    };
    const baseConfig = super.getConfig();
    delete baseConfig['cell'];
    Object.assign(config, baseConfig);
    return config;
  }

  /** @nocollapse */
  static fromConfig<T extends tf.serialization.Serializable>(
    cls: tf.serialization.SerializableConstructor<T>,
    config: tf.serialization.ConfigDict,
  ): T {
    if (config['implmentation'] === 0) {
      config['implementation'] = 1;
    }
    return new cls(config);
  }
}

tf.serialization.registerClass(AttentionLstm);
