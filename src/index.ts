import { type InferenceSession, Tensor } from 'onnxruntime-web'

export interface FaceObject {
  rect: [number, number, number, number]
  landmarks: [[number, number], [number, number], [number, number], [number, number], [number, number]]
  prob: number
}

const generateAnchors = (baseSize: number, ratios: [number], scales: [number, number]) => {
  const numRatio = ratios.length
  const numScale = scales.length

  const anchors: Array<[number, number, number, number]> = []

  const cx = baseSize * 0.5
  const cy = baseSize * 0.5

  for (let i = 0; i < numRatio; i++) {
    const ar = ratios[i]

    const rW = Math.round(baseSize / Math.sqrt(ar))
    const rH = Math.round(rW * ar)

    for (let j = 0; j < numScale; j++) {
      const scale = scales[j]

      const rsW = rW * scale * 0.5
      const rsH = rH * scale * 0.5

      anchors[i * numScale + j] = [cx - rsW, cy - rsH, cx + rsW, cy + rsH]
    }
  }

  return anchors
}

const generateProposals = async (anchors: Array<[number, number, number, number]>, featStride: number, scoreT: Tensor, bboxT: Tensor, landmarkT: Tensor, probThreshold: number) => {
  const bboxDims = bboxT.dims
  const w = bboxDims[3]
  const h = bboxDims[2]
  const offset = w * h

  const score = (await scoreT.getData()) as Float32Array
  const bbox = (await bboxT.getData()) as Float32Array
  const landmark = (await landmarkT.getData()) as Float32Array

  const faces: FaceObject[] = []

  const numAnchors = anchors.length

  for (let q = 0; q < numAnchors; q++) {
    const anchor = anchors[q]

    const scoreOffset = (q + numAnchors) * offset
    const bboxOffset = q * 4 * offset
    const landmarkOffset = q * 10 * offset

    let anchorY = anchor[1]

    const anchorW = anchor[2] - anchor[0]
    const anchorH = anchor[3] - anchor[1]

    for (let i = 0; i < h; i++) {
      let anchorX = anchor[0]

      for (let j = 0; j < w; j++) {
        const index = i * w + j

        const prob = score[scoreOffset + index]

        if (prob >= probThreshold) {
          const dx = bbox[bboxOffset + index + offset * 0]
          const dy = bbox[bboxOffset + index + offset * 1]
          const dw = bbox[bboxOffset + index + offset * 2]
          const dh = bbox[bboxOffset + index + offset * 3]

          const cx = anchorX + anchorW * 0.5
          const cy = anchorY + anchorH * 0.5

          const pbCx = cx + anchorW * dx
          const pbCy = cy + anchorH * dy

          const pbW = anchorW * Math.exp(dw)
          const pbH = anchorH * Math.exp(dh)

          const x0 = pbCx - pbW * 0.5
          const y0 = pbCy - pbH * 0.5
          const x1 = pbCx + pbW * 0.5
          const y1 = pbCy + pbH * 0.5

          const obj: FaceObject = {
            rect: [x0, y0, x1, y1],
            landmarks: [
              [cx + (anchorW + 1) * landmark[landmarkOffset + index + offset * 0], cy + (anchorH + 1) * landmark[landmarkOffset + index + offset * 1]],
              [cx + (anchorW + 1) * landmark[landmarkOffset + index + offset * 2], cy + (anchorH + 1) * landmark[landmarkOffset + index + offset * 3]],
              [cx + (anchorW + 1) * landmark[landmarkOffset + index + offset * 4], cy + (anchorH + 1) * landmark[landmarkOffset + index + offset * 5]],
              [cx + (anchorW + 1) * landmark[landmarkOffset + index + offset * 6], cy + (anchorH + 1) * landmark[landmarkOffset + index + offset * 7]],
              [cx + (anchorW + 1) * landmark[landmarkOffset + index + offset * 8], cy + (anchorH + 1) * landmark[landmarkOffset + index + offset * 9]]
            ],
            prob
          }

          faces.push(obj)
        }

        anchorX += featStride
      }

      anchorY += featStride
    }
  }

  return faces
}

const processStride = async (results: InferenceSession.OnnxValueMapType, faceProposals: FaceObject[], probThreshold: number, stride: number, scales: [number, number]) => {
  const score = results['face_rpn_cls_prob_reshape_stride' + stride]
  const bbox = results['face_rpn_bbox_pred_stride' + stride]
  const landmark = results['face_rpn_landmark_pred_stride' + stride]

  const baseSize = 16
  const featStride = stride
  const anchors = generateAnchors(baseSize, [1], scales)

  faceProposals.push(...(await generateProposals(anchors, featStride, score, bbox, landmark, probThreshold)))
}

const nmsSortedBboxes = (faceObjects: FaceObject[], nmsThreshold: number) => {
  const picked: number[] = []

  const n = faceObjects.length

  const areas = faceObjects.map(obj => (obj.rect[2] - obj.rect[0]) * (obj.rect[3] - obj.rect[1]))

  for (let i = 0; i < n; i++) {
    const a = faceObjects[i]

    let keep = 1
    for (const j of picked) {
      const b = faceObjects[j]

      const interArea = Math.max(0, Math.min(a.rect[2], b.rect[2]) - Math.max(a.rect[0], b.rect[0])) * Math.max(0, Math.min(a.rect[3], b.rect[3]) - Math.max(a.rect[1], b.rect[1]))
      const unionArea = areas[i] + areas[j] - interArea
      if (interArea / unionArea > nmsThreshold) keep = 0
    }

    if (keep) picked.push(i)
  }

  return picked
}

export const createCanvas = (width: number, height: number) => {
  if (typeof OffscreenCanvas !== 'undefined') {
    return new OffscreenCanvas(width, height)
  } else {
    const canvas = document.createElement('canvas')
    canvas.width = width
    canvas.height = height
    return canvas
  }
}

export default class Retinaface {
  public constructor (private readonly session: InferenceSession, public readonly width = 512, public readonly height = 512) {}

  public detect = async (imageData: ImageData, scale = 1, probThreshold = 0.75, nmsThreshold = 0.5): Promise<FaceObject[]> => {
    if (imageData.width !== this.width || imageData.height !== this.height) {
      throw new Error(`image should be ${this.width}x${this.height}`)
    }

    const data = new Float32Array(imageData.data.length / 4 * 3)
    // NCHW
    const len = this.width * this.height
    for (let i = 0; i < len; i++) {
      data[i] = imageData.data[i * 4]
      data[i + len] = imageData.data[i * 4 + 1]
      data[i + len * 2] = imageData.data[i * 4 + 2]
    }

    const results = await this.session.run({ data: new Tensor('float32', data, [1, 3, this.height, this.width]) })

    const faceProposals: FaceObject[] = []
    await processStride(results, faceProposals, probThreshold, 32, [32, 16])
    await processStride(results, faceProposals, probThreshold, 16, [8, 4])
    await processStride(results, faceProposals, probThreshold, 8, [2, 1])

    faceProposals.sort((a, b) => b.prob - a.prob)

    const picked = nmsSortedBboxes(faceProposals, nmsThreshold)

    return picked.map(i => {
      const obj = faceProposals[i]

      const x0 = Math.max(Math.min(obj.rect[0], this.width - 1), 0) / scale
      const y0 = Math.max(Math.min(obj.rect[1], this.height - 1), 0) / scale
      const x1 = Math.max(Math.min(obj.rect[2], this.width - 1), 0) / scale
      const y1 = Math.max(Math.min(obj.rect[3], this.height - 1), 0) / scale

      const face: FaceObject = {
        rect: [x0, y0, x1, y1],
        landmarks: obj.landmarks.map(landmark => [landmark[0] / scale, landmark[1] / scale]) as FaceObject['landmarks'],
        prob: obj.prob
      }
      return face
    })
  }

  public processImage (image: HTMLImageElement, rect?: { left?: number, top?: number, width?: number, height?: number }): [ImageData, number] {
    const canvas = createCanvas(this.width, this.height)
    const ctx = canvas.getContext('2d')! as CanvasRenderingContext2D
    const r = { left: 0, top: 0, width: image.width, height: image.height, ...rect }
    const scale = Math.min(this.width / image.width, this.height / image.height)
    ctx.drawImage(image, r.left, r.top, r.width, r.height, 0, 0, image.width * scale | 0, image.height * scale | 0)
    return [ctx.getImageData(0, 0, this.width, this.height), scale]
  }
}
