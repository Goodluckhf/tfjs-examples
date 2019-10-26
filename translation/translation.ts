/**
 * @license
 * Copyright 2019 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

/**
 * Train a simple LSTM model for character-level language translation.
 * This is based on the Tensorflow.js example at:
 *   https://github.com/tensorflow/tfjs-examples/blob/master/translation/python/translation.py
 *
 * The training data can be downloaded with a command like the following example:
 *   wget http://www.manythings.org/anki/fra-eng.zip
 *
 * Original author: Huan LI <zixia@zixia.net>
 * 2019, https://github.com/huan
 */
import fs from 'fs';
import path from 'path';

import { ArgumentParser } from 'argparse';
import readline from 'readline';
import mkdirp from 'mkdirp';

const { zip } = require('zip-array');
const invertKv = require('invert-kv');

import * as tf from '@tensorflow/tfjs';
import { SaveCallback } from './save-callback';
import {
  PretrainedDecoderMetadata,
  PretrainedEncoderMetadata,
  Seq2seq,
} from './model/seq2seq';
import { SequenceDecoder } from './sequence-decoder';
import { TestBatchCallback } from './test-batch-callback';

let args = {} as any;

async function readData(dataFile: string, maxLength: number) {
  // Vectorize the data.
  const inputTexts: string[] = [];
  const targetTexts: string[] = [];

  const inputCharacters = new Set<string>([
    '\v', // PAD
    '\r', // UNK
  ]);

  const targetCharacters = new Set<string>([
    '\v', // PAD
    '\r', // UNK
    '\t', // SOS
    '\n', // EOS
  ]);

  const fileStream = fs.createReadStream(dataFile);
  const rl = readline.createInterface({
    input: fileStream,
    output: process.stdout,
    terminal: false,
  });

  let lineNumber = 0;
  rl.on('line', line => {
    if (++lineNumber > args.num_samples) {
      rl.close();
      return;
    }

    let [inputText, targetText] = line.split('\t');
    inputText = inputText
      .slice(0, maxLength)
      .toLowerCase()
      .trim();
    // We use "tab" as the "start sequence" character for the targets, and "\n"
    // as "end sequence" character.
    targetText =
      '\t' +
      targetText
        .slice(0, maxLength)
        .toLowerCase()
        .trim() +
      '\n';

    inputTexts.push(inputText.toLowerCase());
    targetTexts.push(targetText.toLowerCase());

    for (const char of inputText) {
      if (!inputCharacters.has(char)) {
        inputCharacters.add(char);
      }
    }
    for (const char of targetText) {
      if (!targetCharacters.has(char)) {
        targetCharacters.add(char);
      }
    }
  });

  await new Promise(r => rl.on('close', r));

  const inputCharacterList = [...inputCharacters];
  const targetCharacterList = [...targetCharacters];

  const numEncoderTokens = inputCharacterList.length;
  const numDecoderTokens = targetCharacterList.length;

  // Math.max() does not work with very large arrays because of the stack limitation
  const maxEncoderSeqLength = inputTexts
    .map(text => text.length)
    .reduceRight((prev, curr) => (curr > prev ? curr : prev), 0);
  const maxDecoderSeqLength = targetTexts
    .map(text => text.length)
    .reduceRight((prev, curr) => (curr > prev ? curr : prev), 0);

  console.log('Number of samples:', inputTexts.length);
  console.log('Number of unique input tokens:', numEncoderTokens);
  console.log('Number of unique output tokens:', numDecoderTokens);
  console.log('Max sequence length for inputs:', maxEncoderSeqLength);
  console.log('Max sequence length for outputs:', maxDecoderSeqLength);

  const inputTokenIndex = inputCharacterList.reduce(
    (prev, curr, idx) => ((prev[curr] = idx), prev),
    {} as { [char: string]: number },
  );
  const targetTokenIndex = targetCharacterList.reduce(
    (prev, curr, idx) => ((prev[curr] = idx), prev),
    {} as { [char: string]: number },
  );

  // Save the token indices to file.
  const metadataJsonPath = path.join(args.artifacts_dir, 'metadata.json');

  if (!fs.existsSync(path.dirname(metadataJsonPath))) {
    mkdirp.sync(path.dirname(metadataJsonPath));
  }

  const metadata = {
    input_token_index: inputTokenIndex,
    target_token_index: targetTokenIndex,
    max_encoder_seq_length: maxEncoderSeqLength,
    max_decoder_seq_length: maxDecoderSeqLength,
    encoder_tokens_length: numEncoderTokens,
    decoder_token_length: numDecoderTokens,
    embedding_dim: args.embedding_dim,
    latent_dim: args.latent_dim,
  };

  fs.writeFileSync(metadataJsonPath, JSON.stringify(metadata));
  console.log('Saved metadata at: ', metadataJsonPath);

  return {
    inputTexts,
    targetTexts,
    maxEncoderSeqLength,
    maxDecoderSeqLength,
    numEncoderTokens,
    numDecoderTokens,
    inputTokenIndex,
    targetTokenIndex,
    reverseTargetCharIndex: invertKv(targetTokenIndex) as {
      [indice: number]: string;
    },
  };
}

function createTrainDataset(
  inputTexts: string[],
  targetTexts: string[],
  sequenceDecoder: SequenceDecoder,
) {
  const inputTextValidation = inputTexts.splice(
    Math.ceil(inputTexts.length * 0.9),
  );
  const targetTextsValidation = targetTexts.splice(
    Math.ceil(targetTexts.length * 0.9),
  );

  function* dataX() {
    for (const [, [inputText, targetText]] of zip(
      inputTexts,
      targetTexts,
    ).entries() as IterableIterator<[number, [string, string]]>) {
      yield sequenceDecoder.getXSample(inputText, targetText);
    }
  }

  function* dataY() {
    for (const [, [, targetText]] of zip(
      inputTexts,
      targetTexts,
    ).entries() as IterableIterator<[number, [string, string]]>) {
      yield sequenceDecoder.getYSample(targetText);
    }
  }

  const xs = tf.data.generator(dataX);
  const ys = tf.data.generator(dataY);
  const trainDs = tf.data
    .zip({ xs, ys })
    .shuffle(args.batch_size * 2)
    .batch(args.batch_size);

  function* dataValidX() {
    for (const [, [inputText, targetText]] of zip(
      inputTextValidation,
      targetTextsValidation,
    ).entries() as IterableIterator<[number, [string, string]]>) {
      yield sequenceDecoder.getXSample(inputText, targetText);
    }
  }

  function* dataValidY() {
    for (const [, [, targetText]] of zip(
      inputTextValidation,
      targetTextsValidation,
    ).entries() as IterableIterator<[number, [string, string]]>) {
      yield sequenceDecoder.getYSample(targetText);
    }
  }

  const xsValid = tf.data.generator(dataValidX);
  const ysValid = tf.data.generator(dataValidY);
  const validDs = tf.data
    .zip({ xs: xsValid, ys: ysValid })
    .batch(args.batch_size);

  return {
    validDs,
    trainDs,
  };
}

async function main() {
  let tfn;
  if (args.gpu) {
    console.log('Using GPU');
    tfn = require('@tensorflow/tfjs-node-gpu');
  } else {
    console.log('Using CPU');
    tfn = require('@tensorflow/tfjs-node');
  }

  const {
    numEncoderTokens,
    numDecoderTokens,
    inputTexts,
    targetTokenIndex,
    maxDecoderSeqLength,
    maxEncoderSeqLength,
    targetTexts,
    inputTokenIndex,
    reverseTargetCharIndex,
  } = await readData(args.data_path, args.max_sequence_length);

  const sequenceDecoder = new SequenceDecoder({
    inputTokenIndex,
    maxDecoderSeqLength,
    maxEncoderSeqLength,
    reverseTargetCharIndex,
    targetTokenIndex,
  });

  const { validDs, trainDs } = createTrainDataset(
    inputTexts,
    targetTexts,
    sequenceDecoder,
  );

  const seq2seq = new Seq2seq({
    embeddingDim: args.embedding_dim,
    latentDim: args.latent_dim,
    inputSequenceLength: maxEncoderSeqLength,
    numEncoderTokens,
    numDecoderTokens,
  });

  const encoder = seq2seq.buildEncoder();
  const decoderMetadata = seq2seq.buildDecoder(encoder.states, encoder.outputs);

  const model = tf.model({
    inputs: [encoder.inputs, decoderMetadata.decoder.inputs],
    outputs: decoderMetadata.decoder.outputs,
    name: 'seq2seqModel',
  });

  // Run training.
  model.compile({
    optimizer: tf.train.adadelta(),
    loss: 'sparseCategoricalCrossentropy',
    metrics: [tf.metrics.sparseCategoricalAccuracy],
  });
  model.summary(120);

  if (args.logDir != null) {
    console.log(
      `To view logs in tensorboard, do:\n` +
        `  tensorboard --logdir ${args.logDir}\n`,
    );
  }

  const pretrainedEncoderMetadata: PretrainedEncoderMetadata = {
    inputs: encoder.inputs,
    outputs: [encoder.outputs, ...encoder.states],
  };

  const pretrainedDecoderMetadata: PretrainedDecoderMetadata = {
    decoder: decoderMetadata.decoder,
  };

  const encoderInferenceModel = seq2seq.buildPretrainedEncoder(
    pretrainedEncoderMetadata,
  );

  const decoderInferenceModel = seq2seq.buildPretrainedDecoder(
    pretrainedDecoderMetadata,
  );

  await model.fitDataset(trainDs, {
    validationData: validDs,
    epochs: args.epochs,
    callbacks: [
      tfn.node.tensorBoard(`./logs/train-${process.pid}`, {
        updateFreq: args.logUpdateFreq,
      }),
      tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 25 }),
      new SaveCallback(
        2,
        `file://${args.artifacts_dir}`,
        encoderInferenceModel,
        decoderInferenceModel,
      ),
      new TestBatchCallback(
        sequenceDecoder,
        encoderInferenceModel,
        decoderInferenceModel,
        {
          everyEpoch: args.test_every_epoch,
          examplesLength: 1,
          targetBeginIndex: targetTokenIndex['\t'],
          testTargetData: targetTexts,
          testInputData: inputTexts,
        },
      ),
    ],
  });

  const encoderModel = seq2seq.buildPretrainedEncoder(
    pretrainedEncoderMetadata,
  );
  const decoderModel = seq2seq.buildPretrainedDecoder(
    pretrainedDecoderMetadata,
  );

  await model.save(`file://${args.artifacts_dir}`, {
    trainableOnly: true,
    includeOptimizer: true,
  });
  await encoderModel.save(`file://${args.artifacts_dir}/encoder`, {
    trainableOnly: true,
  });
  await decoderModel.save(`file://${args.artifacts_dir}/decoder`, {
    trainableOnly: true,
  });
  encoderModel.summary(120);
  decoderModel.summary(120);
  // Reverse-lookup token index to decode sequences back to
  // something readable.

  const targetBeginIndex = targetTokenIndex['\t'];

  for (let seqIndex = 0; seqIndex < args.num_test_sentences; seqIndex++) {
    const sampleIndex = Math.floor(Math.random() * inputTexts.length);
    // Take one sequence (part of the training set)
    // for trying out decoding.
    console.log('sampleIndex', sampleIndex);
    console.log('inputTexts', inputTexts.length);
    const [inputSentence] = inputTexts.slice(sampleIndex, sampleIndex + 1);
    const [targetSentence] = targetTexts.slice(sampleIndex, sampleIndex + 1);
    const { encoderInputs: x } = sequenceDecoder.getXSample(
      inputSentence,
      targetSentence,
    );

    // Array to string
    const decodedSentence = await sequenceDecoder.decode(
      x.expandDims(),
      encoderModel,
      decoderModel,
      targetBeginIndex,
    );

    console.log('-');
    console.log('Input sentence:', inputSentence.trim());
    console.log('Target sentence:', targetSentence.trim());
    console.log('Decoded sentence:', decodedSentence.trim());
  }
}

const parser = new ArgumentParser({
  version: '0.0.1',
  addHelp: true,
  description: 'Keras seq2seq translation model training and serialization',
});

parser.addArgument(['data_path'], {
  type: 'string',
  help: 'Path to the training data, e.g., ~/ml-data/fra-eng/fra.txt',
});
parser.addArgument('--batch_size', {
  type: 'int',
  defaultValue: 64,
  help: 'Training batch size.',
});
parser.addArgument('--epochs', {
  type: 'int',
  defaultValue: 200,
  help: 'Number of training epochs.',
});
parser.addArgument('--latent_dim', {
  type: 'int',
  defaultValue: 256,
  help: 'Latent dimensionality of the encoding space.',
});
parser.addArgument('--embedding_dim', {
  type: 'int',
  defaultValue: 32,
  help: 'Latent dimensionality of the encoding space.',
});
parser.addArgument('--test_every_epoch', {
  type: 'int',
  defaultValue: 5,
  help: 'Latent dimensionality of the encoding space.',
});
parser.addArgument('--num_samples', {
  type: 'int',
  defaultValue: 10000,
  help: 'Number of samples to train on.',
});
parser.addArgument('--max_sequence_length', {
  type: 'int',
  defaultValue: 50,
  help: 'Maximum length of sequence.',
});
parser.addArgument('--num_test_sentences', {
  type: 'int',
  defaultValue: 100,
  help: 'Number of example sentences to test at the end of the training.',
});
parser.addArgument('--artifacts_dir', {
  type: 'string',
  defaultValue: '/tmp/translation.keras',
  help: 'Local path for saving the TensorFlow.js artifacts.',
});
parser.addArgument('--logDir', {
  type: 'string',
  help:
    'Optional tensorboard log directory, to which the loss values ' +
    'will be logged during model training.',
});
parser.addArgument('--logUpdateFreq', {
  type: 'string',
  defaultValue: 'epoch',
  optionStrings: ['batch', 'epoch'],
  help:
    'Frequency at which the loss values will be logged to ' + 'tensorboard.',
});
parser.addArgument('--gpu', {
  action: 'storeTrue',
  help: 'Use tfjs-node-gpu to train the model. Requires CUDA/CuDNN.',
});

[args] = parser.parseKnownArgs();
main();
