import { train } from './train.js';
import { detect } from './detect.js';

const command = process.argv[2];

switch (command) {
  case 'train':
    train();
    break;
  case 'detect':
    detect();
    break;
  default:
    console.log('Usage: npm run <train|detect>');
}