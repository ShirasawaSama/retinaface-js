# Retinaface-JS [![npm](https://img.shields.io/npm/v/retinaface)](https://www.npmjs.com/package/retinaface) [![GitHub](https://img.shields.io/github/license/ShirasawaSama/retinaface-js)](LICENSE)

This is a JavaScript implementation of the Retinaface face detection algorithm. It is based on the [Retinaface](https://arxiv.org/abs/1905.00641) paper.

## Screenshots

![screenshot](screenshots/retinaface.jpg)

## Usage

### Installation

```bash
npm install retinaface onnxruntime-web
```

### Example

```typescript
import * as ort from 'onnxruntime-web'
import Retinaface from 'retinaface'

import modelPath from 'retinaface/mnet.25_v2.onnx?url'
import imagePath from './R.jpg'

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/'

const retinaface = new Retinaface(await ort.InferenceSession.create(modelPath), ort.Tensor)

const image = new Image()
image.src = imagePath
await new Promise((resolve, reject) => {
  image.onload = resolve
  image.onerror = reject
})

const [data, scale] = retinaface.processImage(image)
const result = await retinaface.detect(data, scale)

console.log(result)
```

## Author

Shirasawa

## License

[MIT](LICENSE)
