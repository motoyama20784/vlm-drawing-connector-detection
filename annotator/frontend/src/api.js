import axios from 'axios'

const api = axios.create({ baseURL: '/api' })

export const fetchImages = (dir = 'samples') =>
  api.get('/images', { params: { dir } }).then(r => r.data.images)

export const fetchImageUrl = (filename, dir = 'samples') =>
  `/api/images/${encodeURIComponent(filename)}?dir=${encodeURIComponent(dir)}`

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
