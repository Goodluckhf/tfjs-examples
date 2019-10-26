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
import { Tensor } from '@tensorflow/tfjs';
import * as tfc from '@tensorflow/tfjs-core';
import * as math_utils from '@tensorflow/tfjs-layers/dist/utils/math_utils';
import { tidy } from '@tensorflow/tfjs';
import { AttributeError } from '@tensorflow/tfjs-layers/dist/errors';
import { serialization } from '@tensorflow/tfjs';
import * as K from '@tensorflow/tfjs-layers/dist/backend/tfjs_backend';
import { LayerVariable } from '@tensorflow/tfjs-layers/dist/variables';
import { StackedResidualLstmCell } from './stacked-residual-lstm-cell';

export class AttentionLstm extends tf.layers.Layer {
  /** @nocollapse */
  static className = 'AttentionLstm';
  private readonly attentionLayer: LongauAttention;
  private readonly wc: tf.layers.Layer;
  private readonly cell: RNNCell;
  private readonly returnState: boolean;
  private readonly latentDim: number;

  // @ts-ignore
  private stateSpec: InputSpec[];

  private states_: Tensor[];
  private keptStates: Tensor[][];
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
    this.supportsMasking = true;
    // @ts-ignore
    this.stateSpec = null;
    // @ts-ignore
    this.states_ = null;
    this.cell = new StackedResidualLstmCell({
      cells: [new AttentionLstmCell(args), new AttentionLstmCell(args)],
    });
    this.latentDim = args.units;
    this.keptStates = [];
    this.attentionLayer = new LongauAttention({ units: args.units });
    this.wc = tf.layers.dense({ units: args.units, activation: 'tanh' });
  }

  getStates(): Tensor[] {
    if (this.states_ == null) {
      const numStates = Array.isArray(this.cell.stateSize)
        ? this.cell.stateSize.length
        : 1;
      // @ts-ignore
      return math_utils.range(0, numStates).map(x => null);
    } else {
      return this.states_;
    }
  }

  setStates(states: Tensor[]): void {
    this.states_ = states;
  }

  get states(): Tensor[] {
    if (this.states_ == null) {
      const numStates = Array.isArray(this.cell.stateSize)
        ? this.cell.stateSize.length
        : 1;
      const output: Tensor[] = [];
      for (let i = 0; i < numStates; ++i) {
        // @ts-ignore
        output.push(null);
      }
      return output;
    } else {
      return this.states_;
    }
  }

  set states(s: Tensor[]) {
    this.states_ = s;
  }

  computeMask(
    _: Tensor | Tensor[],
    mask?: Tensor | Tensor[],
  ): Tensor | Tensor[] {
    // @ts-ignore
    return tfc.tidy(() => {
      if (Array.isArray(mask)) {
        mask = mask[0];
      }
      const outputMask = this.returnSequences ? mask : null;

      if (this.returnState) {
        const stateMask = this.states.map(() => null);
        return [outputMask].concat(stateMask);
      } else {
        return outputMask;
      }
    });
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
    // костыль
    this.cell.build([null, this.latentDim, stepInputShape[1]]);
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

  resetStates(states?: Tensor | Tensor[], training = false): void {
    tidy(() => {
      if (!this.stateful) {
        throw new AttributeError(
          'Cannot call resetStates() on an RNN Layer that is not stateful.',
        );
      }

      // @ts-ignore
      const batchSize = this.inputSpec[0].shape[0];
      if (batchSize == null) {
        throw new ValueError(
          'If an RNN is stateful, it needs to know its batch size. Specify ' +
            'the batch size of your input tensors: \n' +
            '- If using a Sequential model, specify the batch size by ' +
            'passing a `batchInputShape` option to your first layer.\n' +
            '- If using the functional API, specify the batch size by ' +
            'passing a `batchShape` option to your Input layer.',
        );
      }
      // Initialize state if null.
      if (this.states_ == null) {
        if (Array.isArray(this.cell.stateSize)) {
          this.states_ = this.cell.stateSize.map(dim =>
            tfc.zeros([batchSize, dim]),
          );
        } else {
          this.states_ = [tfc.zeros([batchSize, this.cell.stateSize])];
        }
      } else if (states == null) {
        // Dispose old state tensors.
        tfc.dispose(this.states_);
        // For stateful RNNs, fully dispose kept old states.
        if (this.keptStates != null) {
          tfc.dispose(this.keptStates);
          this.keptStates = [];
        }

        if (Array.isArray(this.cell.stateSize)) {
          this.states_ = this.cell.stateSize.map(dim =>
            tfc.zeros([batchSize, dim]),
          );
        } else {
          this.states_[0] = tfc.zeros([batchSize, this.cell.stateSize]);
        }
      } else {
        if (!Array.isArray(states)) {
          states = [states];
        }
        if (states.length !== this.states_.length) {
          throw new ValueError(
            `Layer ${this.name} expects ${this.states_.length} state(s), ` +
              `but it received ${states.length} state value(s). Input ` +
              `received: ${states}`,
          );
        }

        if (training === true) {
          // Store old state tensors for complete disposal later, i.e., during
          // the next no-arg call to this method. We do not dispose the old
          // states immediately because that BPTT (among other things) require
          // them.
          this.keptStates.push(this.states_.slice());
        } else {
          tfc.dispose(this.states_);
        }

        for (let index = 0; index < this.states_.length; ++index) {
          const value = states[index];
          const dim = Array.isArray(this.cell.stateSize)
            ? this.cell.stateSize[index]
            : this.cell.stateSize;
          const expectedShape = [batchSize, dim];
          if (!util.arraysEqual(value.shape, expectedShape)) {
            throw new ValueError(
              `State ${index} is incompatible with layer ${this.name}: ` +
                `expected shape=${expectedShape}, received shape=${value.shape}`,
            );
          }
          this.states_[index] = value;
        }
      }
      this.states_ = this.states_.map(state => tfc.keep(state.clone()));
    });
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

  getInitialState(inputs: Tensor): Tensor[] {
    return tidy(() => {
      // Build an all-zero tensor of shape [samples, outputDim].
      // [Samples, timeSteps, inputDim].
      let initialState = tfc.zeros(inputs.shape);
      // [Samples].
      initialState = tfc.sum(initialState, [1, 2]);
      initialState = K.expandDims(initialState); // [Samples, 1].

      if (Array.isArray(this.cell.stateSize)) {
        return this.cell.stateSize.map(dim =>
          dim > 1 ? K.tile(initialState, [1, dim]) : initialState,
        );
      } else {
        return this.cell.stateSize > 1
          ? [K.tile(initialState, [1, this.cell.stateSize])]
          : [initialState];
      }
    });
  }

  get trainableWeights(): LayerVariable[] {
    if (!this.trainable) {
      return [];
    }
    // Porting Note: In TypeScript, `this` is always an instance of `Layer`.
    return [
      ...this.cell.trainableWeights,
      ...this.attentionLayer.trainableWeights,
      ...this.wc.trainableWeights,
    ];
  }

  get nonTrainableWeights(): LayerVariable[] {
    // Porting Note: In TypeScript, `this` is always an instance of `Layer`.
    if (!this.trainable) {
      return [
        ...this.cell.weights,
        ...this.attentionLayer.weights,
        ...this.wc.weights,
      ];
    }
    return [
      ...this.cell.nonTrainableWeights,
      ...this.attentionLayer.nonTrainableWeights,
      ...this.wc.nonTrainableWeights,
    ];
  }

  setFastWeightInitDuringBuild(value: boolean) {
    super.setFastWeightInitDuringBuild(value);
    if (this.cell != null) {
      this.cell.setFastWeightInitDuringBuild(value);
    }
  }

  private getRNNConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      returnSequences: this.returnSequences,
      returnState: this.returnState,
      stateful: this.stateful,
    };
    const cellConfig = this.cell.getConfig();
    config['cell'] = {
      className: this.cell.getClassName(),
      config: cellConfig,
    } as serialization.ConfigDictValue;
    return config;
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
    const rnnConfig = this.getRNNConfig();
    delete baseConfig['cell'];
    return { ...config, ...rnnConfig, ...baseConfig };
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
