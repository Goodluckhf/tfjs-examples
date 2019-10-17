/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
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

import * as loader from './loader';
import * as ui from './ui';
import { SequenceDecoder } from './sequence-decoder';

const HOSTED_URLS = {
  model:
    'https://storage.googleapis.com/tfjs-models/tfjs/translation_en_fr_v1/model.json',
  metadata:
    'https://storage.googleapis.com/tfjs-models/tfjs/translation_en_fr_v1/metadata.json',
};

const LOCAL_URLS = {
  encoder: 'http://localhost:1235/resources/encoder/model.json',
  decoder: 'http://localhost:1235/resources/decoder/model.json',
  model: 'http://localhost:1235/resources/model.json',
  metadata: 'http://localhost:1235/resources/metadata.json',
};

class Translator {
  /**
   * Initializes the Translation demo.
   */
  async init(urls) {
    this.urls = urls;
    this.encoderModel = await loader.loadHostedPretrainedModel(urls.encoder);
    this.decoderModel = await loader.loadHostedPretrainedModel(urls.decoder);
    await this.loadMetadata();
    this.sequenceDecoder = new SequenceDecoder({
      reverseTargetCharIndex: this.reverseTargetCharIndex,
      maxEncoderSeqLength: this.maxEncoderSeqLength,
      maxDecoderSeqLength: this.maxDecoderSeqLength,
      inputTokenIndex: this.inputTokenIndex,
      targetTokenIndex: this.targetTokenIndex,
    });
    return this;
  }

  async loadMetadata() {
    const translationMetadata = await loader.loadHostedMetadata(
      this.urls.metadata,
    );
    this.maxDecoderSeqLength = translationMetadata['max_decoder_seq_length'];
    this.maxEncoderSeqLength = translationMetadata['max_encoder_seq_length'];
    console.log('maxDecoderSeqLength = ' + this.maxDecoderSeqLength);
    console.log('maxEncoderSeqLength = ' + this.maxEncoderSeqLength);
    this.numDecoderTokens = translationMetadata['decoder_token_length'];
    this.numEncoderTokens = translationMetadata['encoder_tokens_length'];
    this.inputTokenIndex = translationMetadata['input_token_index'];
    this.targetTokenIndex = translationMetadata['target_token_index'];
    this.latentDim = translationMetadata['latent_dim'];
    this.embeddingDim = translationMetadata['embedding_dim'];
    this.reverseTargetCharIndex = Object.keys(this.targetTokenIndex).reduce(
      (obj, key) => ((obj[this.targetTokenIndex[key]] = key), obj),
      {},
    );
  }

  /** Translate the given English sentence into French. */
  translate(inputSentence) {
    const { encoderInputs: inputSeq } = this.sequenceDecoder.getXSample(
      inputSentence.toLowerCase(),
      '',
    );

    return this.sequenceDecoder.decode(
      inputSeq.expandDims(),
      this.encoderModel,
      this.decoderModel,
      this.targetTokenIndex['\t'],
    );
  }
}

/**
 * Loads the pretrained model and metadata, and registers the translation
 * function with the UI.
 */
async function setupTranslator() {
  if (await loader.urlExists(HOSTED_URLS.model)) {
    ui.status('Model available: ' + HOSTED_URLS.model);
    const button = document.getElementById('load-pretrained-remote');
    button.addEventListener('click', async () => {
      const translator = await new Translator().init(HOSTED_URLS);
      ui.setTranslationFunction(x => translator.translate(x));
      ui.setEnglish('Go.', x => translator.translate(x));
    });
    button.style.display = 'inline-block';
  }

  if (
    (await loader.urlExists(LOCAL_URLS.encoder)) &&
    (await loader.urlExists(LOCAL_URLS.decoder))
  ) {
    ui.status('Encoder model available: ' + LOCAL_URLS.encoder);
    ui.status('Decoder model available: ' + LOCAL_URLS.decoder);
    const button = document.getElementById('load-pretrained-local');
    button.addEventListener('click', async () => {
      const translator = await new Translator().init(LOCAL_URLS);
      ui.setTranslationFunction(x => translator.translate(x));
      await ui.setEnglish('Go.', x => translator.translate(x));
    });
    button.style.display = 'inline-block';
  }

  ui.status('Standing by.');
}

setupTranslator();
