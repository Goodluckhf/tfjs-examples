import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import { promisify } from 'util';
import readline from 'readline';
import './attention-lstm';
import './attention-lstm-cell';
import './model/longau-attention';
const invertKv = require('invert-kv');
import { SequenceDecoder } from './sequence-decoder';
import { Tokenizer } from './tokenizer';

const readFileAsync = promisify(fs.readFile);
(async () => {
  // @ts-ignore
  const encoder = await tf.loadLayersModel(
    'file://resources/encoder/model.json',
  );
  // @ts-ignore
  const decoder = await tf.loadLayersModel(
    'file://resources/decoder/model.json',
  );
  // @ts-ignore
  const metadata = JSON.parse(
    await readFileAsync('./resources/metadata.json', { encoding: 'utf-8' }),
  );

  const sequenceDecoder = new SequenceDecoder({
    reverseTargetCharIndex: invertKv(metadata.target_token_index) as {
      [indice: number]: string;
    },
    maxEncoderSeqLength: metadata.max_encoder_seq_length,
    maxDecoderSeqLength: metadata.max_decoder_seq_length,
    inputTokenIndex: metadata.input_token_index,
    targetTokenIndex: metadata.target_token_index,
  });

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  rl.on('line', async input => {
    if (input.length === 0) {
      return;
    }
    if (input.length > metadata.max_encoder_seq_length) {
      console.warn(
        'input.length > metadata.max_encoder_seq_length',
        metadata.max_encoder_seq_length,
      );
    }

    const tokenizer = new Tokenizer([]);

    const { encoderInputs: x } = sequenceDecoder.getXSample(
      tokenizer.tokenize(input.toLowerCase()),
      [],
    );

    const answer = await sequenceDecoder.decode(
      x.expandDims(),
      encoder,
      decoder,
      metadata.target_token_index['\t'],
    );
    console.log(`B: ${answer.join(' ')}`);
  });
  console.log('Ready');
})();
