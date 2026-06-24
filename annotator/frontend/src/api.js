import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const fetchImages = (dir = 'samples') =>
  api.get('/images', { params: { dir } }).then(r => r.data.images)

export const fetchImageUrl = (filename, dir = 'samples', area = 'original') =>
  `/api/images/${encodeURIComponent(filename)}?dir=${encodeURIComponent(dir)}&area=${encodeURIComponent(area)}`

export const fetchAnnotation = (filename) =>
  api.get(`/annotations/${encodeURIComponent(filename)}`).then(r => r.data)

export const saveAnnotation = (filename, data) =>
  api.post(`/annotations/${encodeURIComponent(filename)}`, data).then(r => r.data)

export const inferBbox = (image, dir, bbox) =>
  api.post('/infer', { image, dir, bbox }).then(r => r.data)

export const fetchDirs = () =>
  api.get('/dirs').then(r => r.data.dirs)

export const fetchStatus = (dir) =>
  api.get('/status', { params: { dir } }).then(r => r.data.images)

export const fetchMaskingFonts = () =>
  api.get('/masking/fonts').then(r => r.data.fonts)

export const fetchMaskingStatus = (dir) =>
  api.get('/masking/status', { params: { dir } }).then(r => r.data.images)

export const fetchMaskingBboxes = (filename, dir) =>
  api.get('/masking/bboxes', { params: { filename, dir } }).then(r => r.data.bboxes)

export const applyMasking = (filename, dir, bboxes, fontName = null) =>
  api.post('/masking/apply', { filename, dir, bboxes, font_name: fontName }).then(r => r.data)

export const fetchResultsList = (dir) =>
  api.get('/results/list', { params: { dir } }).then(r => r.data)

export const fetchResultsEvaluate = (dir, image, resultFile = null, iouThreshold = 0.3) =>
  api.get('/results/evaluate', {
    params: {
      dir, image,
      ...(resultFile ? { result_file: resultFile } : {}),
      iou_threshold: iouThreshold,
    },
  }).then(r => r.data)
