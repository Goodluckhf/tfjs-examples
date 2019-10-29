export class Tokenizer {
  private readonly tokens: Set<string>;

  constructor(defaultTokens: string[]) {
    this.tokens = new Set<string>([...defaultTokens]);
  }

  removeLowFrequency(sentences: string[][], count: number) {
    const freq = sentences
      .reduce((arr, sentence) => [...arr, ...sentence], [])
      .reduce((freq, word) => {
        if (freq.has(word)) {
          freq.set(word, freq.get(word) + 1);
          return freq;
        }

        freq.set(word, 1);
        return freq;
      }, new Map());

    [...freq.entries()]
      .reduce(
        (set, [word, freq]) => {
          if (freq < count) {
            set.add(word);
          }
          return set;
        },
        new Set() as Set<string>,
      )
      .forEach(token => this.tokens.delete(token));
  }

  tokenize(sentence: string): string[] {
    const tokens = sentence
      .replace(/([^a-zа-яё0-9<>])/gi, ' $&')
      .replace(/[ ]+/g, ' ')
      .split(' ');

    tokens.forEach(t => {
      this.tokens.add(t);
    });

    return tokens;
  }

  get uniqueTokens() {
    return this.tokens;
  }
}
