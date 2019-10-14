import { Callback } from '@tensorflow/tfjs-layers';

export class SaveCallback extends Callback {
  private readonly file: string;
  private readonly everyEpoch: number;

  constructor(everyEpoch = 1, file: string) {
    super();
    this.file = file;
    this.everyEpoch = everyEpoch;
  }

  async onEpochEnd(epoch: number) {
    if (epoch === 0 || epoch % this.everyEpoch !== 0) {
      return;
    }
    console.warn('Saving model...');
    await this.model.save(this.file);
  }
}
