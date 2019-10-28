export class Tokenizer {
  private readonly tokens: Set<string>;

  constructor(defaultTokens: string[]) {
    this.tokens = new Set<string>([...defaultTokens]);
  }

  tokenize(sentence: string): string[] {
    const tokens = sentence
      .replace(/([^a-zа-я0-9<>])/gi, ' $&')
      .replace(/[ ]/g, ' ')
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
