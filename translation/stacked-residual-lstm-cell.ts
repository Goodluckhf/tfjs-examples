import {
  RNNCell,
  StackedRNNCellsArgs,
} from '@tensorflow/tfjs-layers/dist/layers/recurrent';
import * as tf from '@tensorflow/tfjs';
import { Kwargs } from '@tensorflow/tfjs-layers/dist/types';
import { nameScope } from '@tensorflow/tfjs-layers/dist/common';
import { batchGetValue } from '@tensorflow/tfjs-layers/dist/variables';
import { deserialize } from '@tensorflow/tfjs-layers/dist/layers/serialization';

export class StackedResidualLstmCell extends RNNCell {
  /** @nocollapse */
  static className = 'StackedResidualLstmCell';
  protected cells: RNNCell[];

  constructor(args: StackedRNNCellsArgs) {
    super(args);
    this.cells = args.cells;
  }

  call(inputs: tf.Tensor[], kwargs: Kwargs): tf.Tensor | tf.Tensor[] {
    return tf.tidy(() => {
      let [cellInputs, ...states] = inputs as tf.Tensor[];

      this.cells.forEach((cell, index) => {
        const output = cell.call(
          [cellInputs, ...states],
          kwargs,
        ) as tf.Tensor[];
        states = output.slice(1);
        if (index > 0) {
          cellInputs = cellInputs.add(output[0]);
        } else {
          cellInputs = output[0];
        }
      });

      return [cellInputs, ...states];
    });
  }

  public build(inputShape: tf.Shape[]): void {
    this.cells.forEach((cell, i) => {
      nameScope(`RNNCell_${i}`, () => {
        if (i === 0) {
          // @ts-ignore
          cell.build([inputShape[0], inputShape[1] + inputShape[2]]);
        } else {
          cell.build([inputShape[0], inputShape[1]]);
        }
      });
    });
    this.built = true;
  }

  getConfig(): tf.serialization.ConfigDict {
    const cellConfigs: tf.serialization.ConfigDict[] = [];
    for (const cell of this.cells) {
      cellConfigs.push({
        className: cell.getClassName(),
        config: cell.getConfig(),
      });
    }
    const config: tf.serialization.ConfigDict = { cells: cellConfigs };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }

  /** @nocollapse */
  static fromConfig<T extends tf.serialization.Serializable>(
    cls: tf.serialization.SerializableConstructor<T>,
    config: tf.serialization.ConfigDict,
    customObjects = {} as tf.serialization.ConfigDict,
  ): T {
    const cells: RNNCell[] = [];
    for (const cellConfig of config['cells'] as tf.serialization.ConfigDict[]) {
      cells.push(deserialize(cellConfig, customObjects) as RNNCell);
    }
    return new cls({ cells });
  }

  get trainableWeights(): tf.LayerVariable[] {
    if (!this.trainable) {
      return [];
    }
    const weights: tf.LayerVariable[] = [];
    for (const cell of this.cells) {
      weights.push(...cell.trainableWeights);
    }
    return weights;
  }

  get nonTrainableWeights(): tf.LayerVariable[] {
    const weights: tf.LayerVariable[] = [];
    for (const cell of this.cells) {
      weights.push(...cell.nonTrainableWeights);
    }
    if (!this.trainable) {
      const trainableWeights: tf.LayerVariable[] = [];
      for (const cell of this.cells) {
        trainableWeights.push(...cell.trainableWeights);
      }
      return trainableWeights.concat(weights);
    }
    return weights;
  }

  /**
   * Retrieve the weights of a the model.
   *
   * @returns A flat `Array` of `tf.Tensor`s.
   */
  getWeights(): tf.Tensor[] {
    const weights: tf.LayerVariable[] = [];
    for (const cell of this.cells) {
      weights.push(...cell.weights);
    }
    return batchGetValue(weights);
  }

  get stateSize() {
    return this.cells[0].stateSize;
  }
}
tf.serialization.registerClass(StackedResidualLstmCell);
