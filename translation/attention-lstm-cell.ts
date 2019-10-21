import { DataType, Tensor, tidy, serialization } from '@tensorflow/tfjs-core';
import * as tf from '@tensorflow/tfjs';
import { Kwargs } from '@tensorflow/tfjs-layers/dist/types';
import {
  Activation,
  getActivation,
  serializeActivation,
} from '@tensorflow/tfjs-layers/dist/activations';
import {
  getInitializer,
  Initializer,
  Ones,
  serializeInitializer,
} from '@tensorflow/tfjs-layers/dist/initializers';
import {
  Constraint,
  getConstraint,
  serializeConstraint,
} from '@tensorflow/tfjs-layers/dist/constraints';
import {
  getRegularizer,
  Regularizer,
  serializeRegularizer,
} from '@tensorflow/tfjs-layers/dist/regularizers';
import { LayerVariable } from '@tensorflow/tfjs-layers/dist/variables';
import { assertPositiveInteger } from '@tensorflow/tfjs-layers/dist/utils/generic_utils';
import * as math_utils from '@tensorflow/tfjs-layers/dist/utils/math_utils';
import { Shape } from '@tensorflow/tfjs-layers/dist/keras_format/common';
import { getExactlyOneShape } from '@tensorflow/tfjs-layers/dist/utils/types_utils';
import * as K from '@tensorflow/tfjs-layers/dist/backend/tfjs_backend';
import { ValueError } from '@tensorflow/tfjs-layers/dist/errors';
import {
  LSTMCellLayerArgs,
  RNNCell,
} from '@tensorflow/tfjs-layers/dist/layers/recurrent';

function generateDropoutMask(
  ones: () => Tensor,
  rate: number,
  training: boolean | undefined = undefined,
  count = 1,
): Tensor | Tensor[] {
  function droppedInputs(): Tensor {
    return K.dropout(ones(), rate);
  }
  if (count > 1) {
    const mask: Tensor[] = [];
    for (let i = 0; i < count; i++) {
      mask.push(K.inTrainPhase(droppedInputs, ones, training));
    }
    return mask.map(m => tf.keep(m.clone()));
  } else {
    return tf.keep(K.inTrainPhase(droppedInputs, ones, training).clone());
  }
}

export class AttentionLstmCell extends RNNCell {
  /** @nocollapse */
  static className = 'AttentionLstmCell';
  readonly units: number;
  readonly activation: Activation;
  readonly recurrentActivation: Activation;
  readonly useBias: boolean;

  readonly kernelInitializer: Initializer;
  readonly recurrentInitializer: Initializer;
  readonly biasInitializer: Initializer;
  readonly unitForgetBias: boolean | undefined;

  readonly kernelConstraint: Constraint;
  readonly recurrentConstraint: Constraint;
  readonly biasConstraint: Constraint;

  readonly kernelRegularizer: Regularizer;
  readonly recurrentRegularizer: Regularizer;
  readonly biasRegularizer: Regularizer;

  readonly dropout: number;
  readonly recurrentDropout: number;

  readonly stateSize: number[];
  readonly implementation: number;

  readonly DEFAULT_ACTIVATION = 'tanh';
  readonly DEFAULT_RECURRENT_ACTIVATION = 'hardSigmoid';
  readonly DEFAULT_KERNEL_INITIALIZER = 'glorotNormal';
  readonly DEFAULT_RECURRENT_INITIALIZER = 'orthogonal';

  readonly DEFAULT_BIAS_INITIALIZER = 'zeros';

  // @ts-ignore
  kernel: LayerVariable;
  // @ts-ignore
  recurrentKernel: LayerVariable;
  // @ts-ignore
  bias: LayerVariable;

  constructor(args: LSTMCellLayerArgs) {
    super(args);

    this.units = args.units;
    assertPositiveInteger(this.units, 'units');
    this.activation = getActivation(
      args.activation === undefined ? this.DEFAULT_ACTIVATION : args.activation,
    );
    this.recurrentActivation = getActivation(
      args.recurrentActivation === undefined
        ? this.DEFAULT_RECURRENT_ACTIVATION
        : args.recurrentActivation,
    );
    this.useBias = args.useBias == null ? true : args.useBias;

    this.kernelInitializer = getInitializer(
      args.kernelInitializer || this.DEFAULT_KERNEL_INITIALIZER,
    );
    this.recurrentInitializer = getInitializer(
      args.recurrentInitializer || this.DEFAULT_RECURRENT_INITIALIZER,
    );

    this.biasInitializer = getInitializer(
      args.biasInitializer || this.DEFAULT_BIAS_INITIALIZER,
    );
    this.unitForgetBias = args.unitForgetBias;

    // @ts-ignore
    this.kernelRegularizer = getRegularizer(args.kernelRegularizer);
    // @ts-ignore
    this.recurrentRegularizer = getRegularizer(args.recurrentRegularizer);
    // @ts-ignore
    this.biasRegularizer = getRegularizer(args.biasRegularizer);

    // @ts-ignore
    this.kernelConstraint = getConstraint(args.kernelConstraint);
    // @ts-ignore
    this.recurrentConstraint = getConstraint(args.recurrentConstraint);
    // @ts-ignore
    this.biasConstraint = getConstraint(args.biasConstraint);

    this.dropout = math_utils.min([
      1,
      math_utils.max([0, args.dropout == null ? 0 : args.dropout]),
    ]);
    this.recurrentDropout = math_utils.min([
      1,
      math_utils.max([
        0,
        args.recurrentDropout == null ? 0 : args.recurrentDropout,
      ]),
    ]);
    // @ts-ignore
    this.implementation = args.implementation;
    this.stateSize = [this.units, this.units];
    // @ts-ignore
    this.dropoutMask = null;
    // @ts-ignore
    this.recurrentDropoutMask = null;
  }

  public build(inputShape: Shape | Shape[]): void {
    inputShape = getExactlyOneShape(inputShape);
    const inputDim = inputShape[inputShape.length - 1];
    this.kernel = this.addWeight(
      'attentionLstmKernel',
      [inputDim, this.units * 4],
      // @ts-ignore
      null,
      this.kernelInitializer,
      this.kernelRegularizer,
      true,
      this.kernelConstraint,
    );
    this.recurrentKernel = this.addWeight(
      'attentionLstmRecurrentKernel',
      [this.units, this.units * 4],
      // @ts-ignore
      null,
      this.recurrentInitializer,
      this.recurrentRegularizer,
      true,
      this.recurrentConstraint,
    );
    let biasInitializer: Initializer;
    if (this.useBias) {
      if (this.unitForgetBias) {
        const capturedBiasInit = this.biasInitializer;
        const capturedUnits = this.units;
        biasInitializer = new (class CustomInit extends Initializer {
          /** @nocollapse */
          static className = 'CustomInit';
          // @ts-ignore
          apply(shape: Shape, dtype?: DataType): Tensor {
            // TODO(cais): More informative variable names?
            const bI = capturedBiasInit.apply([capturedUnits]);
            const bF = new Ones().apply([capturedUnits]);
            const bCAndH = capturedBiasInit.apply([capturedUnits * 2]);
            return K.concatAlongFirstAxis(
              K.concatAlongFirstAxis(bI, bF),
              bCAndH,
            );
          }
        })();
      } else {
        biasInitializer = this.biasInitializer;
      }
      this.bias = this.addWeight(
        'attentionLstmBias',
        [this.units * 4],
        // @ts-ignore
        null,
        biasInitializer,
        this.biasRegularizer,
        true,
        this.biasConstraint,
      );
    } else {
      // @ts-ignore
      this.bias = null;
    }
    // Porting Notes: Unlike the PyKeras implementation, we perform slicing
    //   of the weights and bias in the call() method, at execution time.
    this.built = true;
  }

  call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[] {
    return tidy(() => {
      const training = kwargs['training'] == null ? false : kwargs['training'];
      inputs = inputs as Tensor[];
      if (inputs.length !== 3) {
        throw new ValueError(
          `LSTMCell expects 3 input Tensors (inputs, h, c), got ` +
            `${inputs.length}.`,
        );
      }
      let hTMinus1 = inputs[1]; // Previous memory state.
      const cTMinus1 = inputs[2]; // Previous carry state.
      inputs = inputs[0];
      if (0 < this.dropout && this.dropout < 1 && this.dropoutMask == null) {
        this.dropoutMask = generateDropoutMask(
          () => tf.onesLike(inputs as Tensor),
          this.dropout,
          training,
          4,
        ) as Tensor[];
      }
      if (
        0 < this.recurrentDropout &&
        this.recurrentDropout < 1 &&
        this.recurrentDropoutMask == null
      ) {
        this.recurrentDropoutMask = generateDropoutMask(
          () => tf.onesLike(hTMinus1),
          this.recurrentDropout,
          training,
          4,
        ) as Tensor[];
      }
      const dpMask = this.dropoutMask as [Tensor, Tensor, Tensor, Tensor];
      const recDpMask = this.recurrentDropoutMask as [
        Tensor,
        Tensor,
        Tensor,
        Tensor,
      ];

      // Note: For superior performance, TensorFlow.js always uses
      // implementation 2 regardless of the actual value of
      // config.implementation.
      let i: Tensor;
      let f: Tensor;
      let c: Tensor;
      let o: Tensor;
      if (0 < this.dropout && this.dropout < 1) {
        inputs = tf.mul(inputs, dpMask[0]);
      }
      let z = K.dot(inputs, this.kernel.read());
      if (0 < this.recurrentDropout && this.recurrentDropout < 1) {
        hTMinus1 = tf.mul(hTMinus1, recDpMask[0]);
      }
      z = tf.add(z, K.dot(hTMinus1, this.recurrentKernel.read()));
      if (this.useBias) {
        z = K.biasAdd(z, this.bias.read());
      }

      const [z0, z1, z2, z3] = tf.split(z, 4, z.rank - 1);

      i = this.recurrentActivation.apply(z0);
      f = this.recurrentActivation.apply(z1);
      c = tf.add(tf.mul(f, cTMinus1), tf.mul(i, this.activation.apply(z2)));
      o = this.recurrentActivation.apply(z3);

      const h = tf.mul(o, this.activation.apply(c));
      // TODO(cais): Add use_learning_phase flag properly.
      return [h, h, c];
    });
  }

  getConfig(): serialization.ConfigDict {
    const config: serialization.ConfigDict = {
      units: this.units,
      activation: serializeActivation(this.activation),
      recurrentActivation: serializeActivation(this.recurrentActivation),
      useBias: this.useBias,
      kernelInitializer: serializeInitializer(this.kernelInitializer),
      recurrentInitializer: serializeInitializer(this.recurrentInitializer),
      biasInitializer: serializeInitializer(this.biasInitializer),
      // @ts-ignore
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
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.registerClass(AttentionLstmCell);
